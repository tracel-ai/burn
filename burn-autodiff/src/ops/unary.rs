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
    type StateForward: Clone + Send + Sync + std::fmt::Debug + 'static;
    type StateBackward: Clone + Send + Sync + std::fmt::Debug + 'static;

    fn forward(
        &self,
        tensor: B::TensorPrimitive<DI>,
        state: Self::StateForward,
    ) -> B::TensorPrimitive<DO>;
    fn backward(
        self,
        tensor: Option<MetadataRef>,
        output: BackwardTensor<B, DO>,
        grads: &mut Gradients<B>,
        state: Self::StateBackward,
    );
    fn execute(
        self,
        tensor: ADTensor<B, DI>,
        state_forward: Self::StateForward,
        state_backward: Self::StateBackward,
    ) -> ADTensor<B, DO> {
        if let Requirement::None = tensor.metadata.requirement {
            return ADTensor::from_unary_ops(
                tensor.metadata.clone(),
                self.forward(tensor.primitive, state_forward),
                tensor.graph,
            );
        }

        let output = ADTensor::from_unary_ops(
            tensor.metadata.clone(),
            self.forward(tensor.primitive, state_forward),
            tensor.graph,
        );
        let ops = UnaryOpsNoCaptureBackward::new(
            tensor.metadata.clone_if_require_grad(),
            output.to_backward(),
            self,
            state_backward,
        );

        output.register_ops(ops)
    }
}

#[derive(new, Debug)]
struct UnaryOpsNoCaptureBackward<B, T, SF, SB, const DI: usize, const DO: usize>
where
    B: Backend,
    SF: Clone + Send + Sync + std::fmt::Debug + 'static,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
    T: UnaryOpsNoCapture<B, DI, DO, StateForward = SF, StateBackward = SB>,
{
    tensor: Option<MetadataRef>,
    output: BackwardTensor<B, DO>,
    ops: T,
    state: SB,
}

impl<B, T, SF, SB, const DI: usize, const DO: usize> Backward<B>
    for UnaryOpsNoCaptureBackward<B, T, SF, SB, DI, DO>
where
    B: Backend,
    SF: Clone + Send + Sync + std::fmt::Debug,
    SB: Clone + Send + Sync + std::fmt::Debug,
    T: UnaryOpsNoCapture<B, DI, DO, StateForward = SF, StateBackward = SB>,
{
    fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
        self.ops
            .backward(self.tensor, self.output, grads, self.state);
    }

    fn metadata(&self) -> MetadataRef {
        self.output.metadata.clone()
    }
}
