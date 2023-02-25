use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::ops::{Backward, MetadataRef, Requirement},
    tensor::{ADTensor, BackwardTensor},
};

/// Unary operation that does not require the input tensor to be collected during the foward pass.
pub trait UnaryOpsNoCapture<B: Backend, S, const DI: usize, const DO: usize>:
    Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn forward(&self, tensor: B::TensorPrimitive<DI>, state: S) -> B::TensorPrimitive<DO>;
    fn backward(
        self,
        tensor: Option<MetadataRef>,
        output: BackwardTensor<B, DO>,
        grads: &mut Gradients<B>,
        state: S,
    );
    fn execute(self, tensor: ADTensor<B, DI>, state: S) -> ADTensor<B, DO> {
        if let Requirement::None = tensor.metadata.requirement {
            return ADTensor::from_unary_ops(
                tensor.metadata.clone(),
                self.forward(tensor.primitive, state),
                tensor.graph,
            );
        }

        let output = ADTensor::from_unary_ops(
            tensor.metadata.clone(),
            self.forward(tensor.primitive, state.clone()),
            tensor.graph,
        );
        let ops = UnaryOpsNoCaptureBackward::new(
            tensor.metadata.clone_if_require_grad(),
            output.to_backward(),
            self,
            state,
        );

        output.register_ops(ops)
    }
}

#[derive(new, Debug)]
struct UnaryOpsNoCaptureBackward<B, T, S, const DI: usize, const DO: usize>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
    T: UnaryOpsNoCapture<B, S, DI, DO>,
{
    tensor: Option<MetadataRef>,
    output: BackwardTensor<B, DO>,
    ops: T,
    state: S,
}

impl<B, T, S, const DI: usize, const DO: usize> Backward<B>
    for UnaryOpsNoCaptureBackward<B, T, S, DI, DO>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug,
    T: UnaryOpsNoCapture<B, S, DI, DO>,
{
    fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
        self.ops
            .backward(self.tensor, self.output, grads, self.state);
    }

    fn metadata(&self) -> MetadataRef {
        self.output.metadata.clone()
    }
}
