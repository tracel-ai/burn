use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::{
        NodeRef, Requirement, {Graph, Step},
    },
    tensor::{ADTensor, BackwardTensor},
};

/// Trait for all operations.
///
/// An operation have a forward and a backward associated type to specify what data are necessary
/// during those steps.
///
/// # Notes
///
/// Concrete types implementing this trait should not have any state, they are only used to dispatch
/// the right function implementation. If data is necessary during the foward or backward pass,
/// they should be declared with the associated types.
pub trait Ops<B, const D: usize, const N: usize>: Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
    B: Backend,
{
    /// Associated type to compute the forward pass.
    type Forward: Clone + Send + Sync + std::fmt::Debug + 'static;
    /// Associated type to compute the backward pass.
    type Backward: Clone + Send + Sync + std::fmt::Debug + 'static;

    /// The forward pass.
    fn forward(&self, state: Self::Forward) -> B::TensorPrimitive<D>;
    /// The backward pass.
    fn backward(
        self,
        nodes: OpsNodes<N>,
        output: BackwardTensor<B, D>,
        grads: &mut Gradients<B>,
        state: Self::Backward,
    );
    /// Run the operation:
    ///
    /// 1. Determine the right grad requirement.
    /// 2. Execute the forward pass.
    /// 3. Create the backward step (if required)
    /// 4. Register the step into the graph (if required)
    /// 5. Returns the tensor.
    fn run(
        self,
        nodes: [NodeRef; N],
        graphs: [Graph<B>; N],
        state_forward: Self::Forward,
        state_backward: Self::Backward,
    ) -> ADTensor<B, D> {
        let requirement = Requirement::from_nodes(&nodes);

        if let Requirement::None = requirement {
            // Free the backward state, so if there are any reference
            // to the same tensors used in the forward pass, inplace
            // operations could be used.
            std::mem::drop(state_backward);

            return ADTensor::from_ops(&nodes, self.forward(state_forward), graphs, requirement);
        }

        let output = ADTensor::from_ops(&nodes, self.forward(state_forward), graphs, requirement);

        let nodes = nodes.map(|node| match node.clone_if_require_grad() {
            Some(node) => OpsNode::Node(node),
            None => OpsNode::Untrack,
        });
        let ops = OpsBackward::new(nodes, output.to_backward(), self, state_backward);

        output.register_ops(ops)
    }
}

pub type OpsNodes<const N: usize> = [OpsNode; N];

#[derive(Debug)]
pub enum OpsNode {
    Node(NodeRef),
    Untrack,
}

impl OpsNode {
    pub fn run<F>(self, func: F)
    where
        F: FnOnce(NodeRef),
    {
        match self {
            Self::Node(node) => func(node),
            Self::Untrack => (),
        }
    }
}

#[derive(new, Debug)]
struct OpsBackward<B, T, SF, SB, const D: usize, const N: usize>
where
    B: Backend,
    T: Ops<B, D, N, Forward = SF, Backward = SB>,
    SF: Clone + Send + Sync + std::fmt::Debug + 'static,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    nodes: OpsNodes<N>,
    output: BackwardTensor<B, D>,
    ops: T,
    state: SB,
}

impl<B, T, SF, SB, const D: usize, const NUM_INPUTS: usize> Step<B>
    for OpsBackward<B, T, SF, SB, D, NUM_INPUTS>
where
    B: Backend,
    T: Ops<B, D, NUM_INPUTS, Forward = SF, Backward = SB>,
    SF: Clone + Send + Sync + std::fmt::Debug + 'static,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn step(self: Box<Self>, grads: &mut Gradients<B>) {
        self.ops
            .backward(self.nodes, self.output, grads, self.state);
    }

    fn node(&self) -> NodeRef {
        self.output.node.clone()
    }
}
