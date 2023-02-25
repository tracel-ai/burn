use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::ops::{Backward, MetadataRef, Requirement},
    tensor::{ADTensor, BackwardTensor},
};

pub trait BinaryOps<B: Backend, const D: usize>: Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
{
    type BackwardState: Clone + Send + Sync + std::fmt::Debug + 'static;

    fn forward(
        &self,
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn backward(
        self,
        lhs: Option<MetadataRef>,
        rhs: Option<MetadataRef>,
        output: BackwardTensor<B, D>,
        grads: &mut Gradients<B>,
        state: Self::BackwardState,
    );
    fn execute(
        self,
        lhs: ADTensor<B, D>,
        rhs: ADTensor<B, D>,
        state: Self::BackwardState,
    ) -> ADTensor<B, D> {
        let output = ADTensor::from_binary_ops(
            lhs.metadata.clone(),
            rhs.metadata.clone(),
            self.forward(lhs.primitive, rhs.primitive),
            lhs.graph,
            rhs.graph,
        );

        if let Requirement::None = output.metadata.requirement {
            return output;
        }

        let ops = BinaryOpsBackward::new(
            lhs.metadata.clone_if_require_grad(),
            rhs.metadata.clone_if_require_grad(),
            output.to_backward(),
            self,
            state,
        );

        output.register_ops(ops)
    }
}

#[derive(new, Debug)]
struct BinaryOpsBackward<B, T, S, const D: usize>
where
    B: Backend,
    T: BinaryOps<B, D, BackwardState = S>,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    lhs: Option<MetadataRef>,
    rhs: Option<MetadataRef>,
    output: BackwardTensor<B, D>,
    ops: T,
    state: S,
}

impl<B, T, S, const D: usize> Backward<B> for BinaryOpsBackward<B, T, S, D>
where
    B: Backend,
    T: BinaryOps<B, D, BackwardState = S>,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
        self.ops
            .backward(self.lhs, self.rhs, self.output, grads, self.state);
    }

    fn metadata(&self) -> MetadataRef {
        self.output.metadata.clone()
    }
}
