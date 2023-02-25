use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::ops::{Backward, MetadataRef, Requirement},
    tensor::{ADTensor, BackwardTensor},
};

/// Unary operation that does not require the input tensor to be collected during the foward pass.
pub trait UnaryOpsNoCapture<B: Backend, const DI: usize, const DO: usize>:
    Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
{
    fn forward(&self, tensor: B::TensorPrimitive<DI>) -> B::TensorPrimitive<DO>;
    fn backward(
        self,
        tensor: Option<MetadataRef>,
        output: BackwardTensor<B, DO>,
        grads: &mut Gradients<B>,
    );
    fn execute(self, tensor: ADTensor<B, DI>) -> ADTensor<B, DO> {
        let output = ADTensor::from_unary_ops(
            tensor.metadata.clone(),
            self.forward(tensor.primitive),
            tensor.graph,
        );

        if let Requirement::None = output.metadata.requirement {
            return output;
        }

        let ops = UnaryOpsNoCaptureBackward::new(
            tensor.metadata.clone_if_require_grad(),
            output.to_backward(),
            self,
        );

        output.register_ops(ops)
    }
}

#[derive(new, Debug)]
struct UnaryOpsNoCaptureBackward<B, T, const DI: usize, const DO: usize>
where
    B: Backend,
    T: UnaryOpsNoCapture<B, DI, DO>,
{
    tensor: Option<MetadataRef>,
    output: BackwardTensor<B, DO>,
    ops: T,
}

impl<B, T, const DI: usize, const DO: usize> Backward<B> for UnaryOpsNoCaptureBackward<B, T, DI, DO>
where
    B: Backend,
    T: UnaryOpsNoCapture<B, DI, DO>,
{
    fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
        self.ops.backward(self.tensor, self.output, grads);
    }

    fn metadata(&self) -> MetadataRef {
        self.output.metadata.clone()
    }
}
