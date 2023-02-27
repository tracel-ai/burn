use std::marker::PhantomData;

use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::{
        NodeRef, Requirement, {Graph, Step},
    },
    tensor::{ADTensor, BackwardTensor},
};

/// Trait for all operatiors.
///
/// # Notes
///
/// Concrete types implementing this trait should not have any state, they are only used to dispatch
/// the right function implementation.
///
/// If a state is necessary during the backward pass,
/// they should be declared with the associated types.
pub trait Backward<B, const D: usize, const N: usize>: Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
    B: Backend,
{
    /// Associated type to compute the backward pass.
    type State: Clone + Send + Sync + std::fmt::Debug + 'static;

    /// The backward pass.
    fn backward(
        self,
        nodes: OpsNodes<N>,
        output: NodeRef,
        grads: &mut Gradients,
        state: Self::State,
    );
    /// Run the operation:
    ///
    /// 1. Determine the right grad requirement.
    /// 2. Create the backward step (if required)
    /// 3. Register the step into the graph (if required)
    /// 4. Returns the tensor.
    fn run(
        self,
        state: Self::State,
        output: B::TensorPrimitive<D>,
        nodes: [NodeRef; N],
        graphs: [Graph; N],
    ) -> ADTensor<B, D> {
        let requirement = Requirement::from_nodes(&nodes);
        let output = ADTensor::from_ops(&nodes, output, graphs, requirement);

        if let Requirement::None = requirement {
            return output;
        }

        let nodes = nodes.map(|node| match node.clone_if_require_grad() {
            Some(node) => OpsNode::Tracked(node, []),
            None => OpsNode::Untrack,
        });
        let ops = OpsStep::new(nodes, output.node.clone(), self, state);

        output.register_ops(ops)
    }
}

pub type OpsNodes<const N: usize> = [OpsNode<(), 0>; N];

/// Node operation.
#[derive(Debug)]
pub enum OpsNode<T, const D: usize> {
    Tracked(NodeRef, [T; D]),
    Untrack,
}

impl<T, const D: usize> OpsNode<T, D> {
    /// Run the backward pass.
    ///
    /// The function should update the node gradient using the [gradients](crate::grads::Gradients) struct.
    pub fn run<F>(self, func: F)
    where
        F: FnOnce(NodeRef, [T; D]),
    {
        match self {
            Self::Tracked(node, args) => func(node, args),
            Self::Untrack => (),
        }
    }
}

impl OpsNode<(), 0> {
    /// Set the required objects for the operation to be executed.
    ///
    /// This is usefull in combination with [duplicate](crate::utils::duplicate) to maximize
    /// inplace tensor operations when possible.
    pub fn requirements<T, const D: usize>(self, items: [Option<T>; D]) -> OpsNode<T, D> {
        match self {
            Self::Tracked(node, _) => OpsNode::Tracked(node, items.map(|item| item.unwrap())),
            Self::Untrack => OpsNode::Untrack,
        }
    }
}

#[derive(new, Debug)]
struct OpsStep<B, T, SB, const D: usize, const N: usize>
where
    B: Backend,
    T: Backward<B, D, N, State = SB>,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    nodes: OpsNodes<N>,
    output: NodeRef,
    ops: T,
    state: SB,
    phantom: PhantomData<B>,
}

impl<B, T, SB, const D: usize, const N: usize> Step for OpsStep<B, T, SB, D, N>
where
    B: Backend,
    T: Backward<B, D, N, State = SB>,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn step(self: Box<Self>, grads: &mut Gradients) {
        self.ops
            .backward(self.nodes, self.output, grads, self.state);
    }

    fn node(&self) -> NodeRef {
        self.output.clone()
    }
}
