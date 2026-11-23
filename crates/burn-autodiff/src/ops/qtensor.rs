use burn_backend::{
    Backend, ExecutionError, TensorData, TensorMetadata, TensorPrimitive,
    ops::{FloatTensorOps, QTensorOps},
    quantization::QuantizationParametersPrimitive,
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{FloatDType, IntDType, QuantScheme, Shape};

use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind, broadcast_shape, unary},
    tensor::AutodiffTensor,
};

/// Which operand of a mixed matmul is the quantized one.
#[derive(Debug, Clone, Copy)]
enum QuantizedSide {
    Lhs,
    Rhs,
}

impl QuantizedSide {
    /// Multiplies on the inner backend, with the quantized operand on this side.
    fn matmul<B: Backend>(
        self,
        float: FloatTensor<B>,
        quantized: QuantizedTensor<B>,
    ) -> TensorPrimitive<B> {
        let float = TensorPrimitive::Float(float);
        let quantized = TensorPrimitive::QFloat(quantized);

        match self {
            Self::Lhs => B::q_matmul(quantized, float),
            Self::Rhs => B::q_matmul(float, quantized),
        }
    }
}

/// Matmul of a float tensor with a quantized one, differentiated with respect to the float
/// operand only.
///
/// The quantized operand is frozen, so the only thing backward needs is the operand itself. It
/// is kept in its packed form and handed back to the inner backend's `q_matmul`, which lets a
/// backend that multiplies packed weights directly do so in both passes. Dequantizing it here
/// instead would hold a dense copy of every quantized weight from the forward pass until
/// backward.
fn matmul_quantized<B: Backend, C: CheckpointStrategy>(
    float: AutodiffTensor<B>,
    quantized: QuantizedTensor<B>,
    side: QuantizedSide,
) -> TensorPrimitive<Autodiff<B, C>> {
    #[derive(Debug)]
    struct MatmulQuantized;

    impl<B: Backend> Backward<B, 1> for MatmulQuantized {
        type State = (QuantizedTensor<B>, QuantizedSide, Shape);

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let (quantized, side, float_shape) = ops.state;

            unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                let dtype: FloatDType = grad.dtype().into();
                let rank = quantized.rank();
                let transposed = B::q_swap_dims(quantized, rank - 2, rank - 1);

                let grad = match side.matmul::<B>(grad, transposed) {
                    TensorPrimitive::Float(grad) => grad,
                    TensorPrimitive::QFloat(grad) => B::dequantize(grad, dtype),
                };

                broadcast_shape::<B>(grad, &float_shape)
            });
        }
    }

    let node = float.node();
    let float_shape = float.primitive.shape();

    let output = match side.matmul::<B>(float.primitive, quantized.clone()) {
        TensorPrimitive::Float(output) => output,
        // The device propagates quantization: the product was quantized, which detaches it
        // just as `quantize` does.
        TensorPrimitive::QFloat(output) => return TensorPrimitive::QFloat(output),
    };

    let output = match MatmulQuantized
        .prepare::<C>([node])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((quantized, side, float_shape), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };

    TensorPrimitive::Float(output)
}

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        B::q_from_data(data, device)
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        // Quantization detaches: the rounding makes it non-differentiable, so
        // the packed tensor starts a new (grad-free) history. Quantization-aware
        // training would need a straight-through estimator here instead.
        B::quantize(
            tensor.primitive,
            scheme,
            QuantizationParametersPrimitive {
                scales: qparams.scales.primitive,
                global: qparams.global.map(|global| global.primitive),
            },
        )
    }

    fn quantize_dynamic(tensor: FloatTensor<Self>, scheme: &QuantScheme) -> QuantizedTensor<Self> {
        B::quantize_dynamic(tensor.primitive, scheme)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        // A quantized tensor carries no autodiff graph, so its float form is a
        // new leaf that does not require gradients. Ops composing it with
        // tracked tensors (e.g. a LoRA adapter over a frozen quantized base)
        // propagate gradients through the tracked side only.
        AutodiffTensor::new(B::dequantize(tensor, dtype))
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(Self::float_matmul(lhs, rhs))
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                matmul_quantized::<B, C>(lhs, rhs, QuantizedSide::Rhs)
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs)) => {
                matmul_quantized::<B, C>(rhs, lhs, QuantizedSide::Lhs)
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                // Neither operand carries a graph, so the product is a leaf.
                match B::q_matmul(TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) {
                    TensorPrimitive::Float(output) => {
                        TensorPrimitive::Float(AutodiffTensor::new(output))
                    }
                    TensorPrimitive::QFloat(output) => TensorPrimitive::QFloat(output),
                }
            }
        }
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        B::q_to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        B::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        B::q_into_data(tensor).await
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

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmax(tensor, dim, out_dtype)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmin(tensor, dim, out_dtype)
    }
}
