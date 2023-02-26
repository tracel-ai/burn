use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::{
        ops::{Backward, Graph, MetadataRef},
        Requirement,
    },
    tensor::{ADTensor, BackwardTensor},
};

pub trait Ops<B, const D: usize, const N: usize>: Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
    B: Backend,
{
    type StateForward: Clone + Send + Sync + std::fmt::Debug + 'static;
    type StateBackward: Clone + Send + Sync + std::fmt::Debug + 'static;

    fn forward(&self, state: Self::StateForward) -> B::TensorPrimitive<D>;
    fn backward(
        self,
        nodes: [Option<MetadataRef>; N],
        output: BackwardTensor<B, D>,
        grads: &mut Gradients<B>,
        state: Self::StateBackward,
    );
    fn run(
        self,
        nodes: [MetadataRef; N],
        graphs: [Graph<B>; N],
        state_forward: Self::StateForward,
        state_backward: Self::StateBackward,
    ) -> ADTensor<B, D> {
        let requirement = Requirement::from_metadata(&nodes);

        if let Requirement::None = requirement {
            // Free the backward state, so if there are any reference
            // to the same tensors used in the forward pass, inplace
            // operations could be used.
            std::mem::drop(state_backward);

            return ADTensor::from_ops(&nodes, self.forward(state_forward), graphs, requirement);
        }

        let output = ADTensor::from_ops(&nodes, self.forward(state_forward), graphs, requirement);

        let nodes = nodes.map(|metadata| metadata.clone_if_require_grad());
        let ops = OpsBackward::new(nodes, output.to_backward(), self, state_backward);

        output.register_ops(ops)
    }
}

#[derive(new, Debug)]
struct OpsBackward<B, T, SF, SB, const D: usize, const NUM_INPUTS: usize>
where
    B: Backend,
    T: Ops<B, D, NUM_INPUTS, StateForward = SF, StateBackward = SB>,
    SF: Clone + Send + Sync + std::fmt::Debug + 'static,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    nodes: [Option<MetadataRef>; NUM_INPUTS],
    output: BackwardTensor<B, D>,
    ops: T,
    state: SB,
}

impl<B, T, SF, SB, const D: usize, const NUM_INPUTS: usize> Backward<B>
    for OpsBackward<B, T, SF, SB, D, NUM_INPUTS>
where
    B: Backend,
    T: Ops<B, D, NUM_INPUTS, StateForward = SF, StateBackward = SB>,
    SF: Clone + Send + Sync + std::fmt::Debug + 'static,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
        self.ops
            .backward(self.nodes, self.output, grads, self.state);
    }

    fn metadata(&self) -> MetadataRef {
        self.output.metadata.clone()
    }
}
