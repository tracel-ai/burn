use alloc::vec::Vec;
use burn_backend::{
    Backend, ExecutionError, TensorData, TensorMetadata,
    ops::QTensorOps,
    quantization::QuantizationParametersPrimitive,
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{FloatDType, IntDType, QuantScheme, Shape};

use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, unary},
    qtensor::AutodiffQTensor,
};

/// Quantizing a float tensor: the gradient passes through unchanged, cast back to the input's
/// precision — a straight-through estimator, so the rounding is never differentiated.
#[derive(Debug)]
struct Quantize;

impl<B: Backend> Backward<B, 1> for Quantize {
    type State = FloatDType;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let input_dtype = ops.state;
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            match FloatDType::from(grad.dtype()) == input_dtype {
                true => grad,
                false => B::float_cast(grad, input_dtype),
            }
        });
    }
}

/// Decoding a quantized tensor: the gradient passes through unchanged.
#[derive(Debug)]
struct Dequantize;

impl<B: Backend> Backward<B, 1> for Dequantize {
    type State = ();

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
    }
}

/// A layout operation a quantized tensor runs as it is stored, named by what its backward undoes.
#[derive(Debug, Clone)]
enum QuantizedLayout<B: Backend> {
    /// Reshaped from this shape.
    Reshape(Shape),
    /// Two dimensions swapped.
    SwapDims(usize, usize),
    /// Axes permuted by this permutation.
    Permute(Vec<usize>),
    /// Flipped along these axes.
    Flip(Vec<usize>),
    /// Moved from this device.
    ToDevice(Device<B>),
}

impl<B: Backend> Backward<B, 1> for QuantizedLayout<B> {
    type State = ();

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| match self {
            Self::Reshape(shape) => B::float_reshape(grad, shape),
            Self::SwapDims(dim1, dim2) => B::float_swap_dims(grad, dim1, dim2),
            Self::Permute(axes) => {
                let mut inverse = alloc::vec![0; axes.len()];
                for (position, axis) in axes.iter().enumerate() {
                    inverse[*axis] = position;
                }
                B::float_permute(grad, &inverse)
            }
            Self::Flip(axes) => B::float_flip(grad, &axes),
            Self::ToDevice(device) => B::float_to_device(grad, &device),
        });
    }
}

impl<B: Backend> QuantizedLayout<B> {
    /// `tensor` laid out by `op`, the node gradients reach it through linked to the input's.
    fn run<C: CheckpointStrategy>(
        self,
        tensor: AutodiffQTensor<B>,
        op: impl FnOnce(QuantizedTensor<B>) -> QuantizedTensor<B>,
    ) -> AutodiffQTensor<B> {
        let node = tensor.node();
        let output = op(tensor.primitive);
        self.prepare::<C>([node])
            .compute_bound()
            .quantized((), output)
    }
}

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        AutodiffQTensor::new(B::q_from_data(data, device))
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let input_dtype = FloatDType::from(tensor.dtype());
        let node = tensor.node();
        let output = B::quantize(
            tensor.primitive,
            scheme,
            QuantizationParametersPrimitive {
                scales: qparams.scales.primitive,
                global: qparams.global.map(|global| global.primitive),
            },
        );
        Quantize
            .prepare::<C>([node])
            .compute_bound()
            .quantized(input_dtype, output)
    }

    fn quantize_dynamic(tensor: FloatTensor<Self>, scheme: &QuantScheme) -> QuantizedTensor<Self> {
        let input_dtype = FloatDType::from(tensor.dtype());
        let node = tensor.node();
        let output = B::quantize_dynamic(tensor.primitive, scheme);
        Quantize
            .prepare::<C>([node])
            .compute_bound()
            .quantized(input_dtype, output)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let node = tensor.node();
        let output = B::dequantize(tensor.primitive, dtype);
        Dequantize
            .prepare::<C>([node])
            .compute_bound()
            .stateless(output)
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        QuantizedLayout::ToDevice(tensor.device())
            .run::<C>(tensor, |tensor| B::q_to_device(tensor, device))
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        QuantizedLayout::Reshape(tensor.shape())
            .run::<C>(tensor, |tensor| B::q_reshape(tensor, shape))
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        B::q_into_data(tensor.primitive).await
    }

    fn q_detach(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        AutodiffQTensor::new(tensor.primitive)
    }

    fn q_is_require_grad(tensor: &QuantizedTensor<Self>) -> bool {
        tensor.is_tracked()
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        QuantizedLayout::SwapDims(dim1, dim2)
            .run::<C>(tensor, |tensor| B::q_swap_dims(tensor, dim1, dim2))
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        QuantizedLayout::Permute(axes.to_vec())
            .run::<C>(tensor, |tensor| B::q_permute(tensor, axes))
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        QuantizedLayout::Flip(axes.to_vec()).run::<C>(tensor, |tensor| B::q_flip(tensor, axes))
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmax(tensor.primitive, dim, out_dtype)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmin(tensor.primitive, dim, out_dtype)
    }
}
