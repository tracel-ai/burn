use super::Backward;
use crate::{
    grads::Gradients,
    graph::{
        NodeRef, Requirement, {Graph, Step},
    },
    tensor::ADTensor,
};
use burn_tensor::backend::Backend;
use std::marker::PhantomData;

/// Operation in preparation.
///
/// There are 3 different modes: 'Init', 'Tracked' and 'UnTracked'.
/// Each mode has its own set of functions to minimize cloning for unused backward states.
#[derive(new)]
pub struct OpsPrep<Backward, B, S, const D: usize, const N: usize, Mode = Init> {
    nodes: [NodeRef; N],
    graphs: [Graph; N],
    requirement: Requirement,
    backward: Backward,
    phantom_backend: PhantomData<B>,
    phantom_state: PhantomData<S>,
    marker: PhantomData<Mode>,
}

pub struct Init;
pub struct Tracked;
pub struct UnTracked;

impl<BO, B, const D: usize, const N: usize> OpsPrep<BO, B, (), D, N, Init>
where
    B: Backend,
    BO: Backward<B, D, N, State = ()>,
{
    /// Prepare an stateless operation.
    pub fn stateless(self, output: <B as Backend>::TensorPrimitive<D>) -> ADTensor<B, D> {
        match self.statefull() {
            OpsKind::Tracked(prep) => prep.finish((), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

impl<BO, B, S, const D: usize, const N: usize> OpsPrep<BO, B, S, D, N, Init>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
    BO: Backward<B, D, N, State = S>,
{
    /// Prepare an operation that requires a state during the backward pass.
    pub fn statefull(self) -> OpsKind<BO, B, S, D, N> {
        match self.requirement.is_none() {
            false => OpsKind::Tracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
            )),
            true => OpsKind::UnTracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
            )),
        }
    }
}

impl<BO, B, S, const D: usize, const N: usize> OpsPrep<BO, B, S, D, N, UnTracked>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
    BO: Backward<B, D, N, State = S>,
{
    /// Finish the preparation of an untracked operation and returns the output tensor.
    pub fn finish(self, output: <B as Backend>::TensorPrimitive<D>) -> ADTensor<B, D> {
        ADTensor::from_parents(
            output,
            &self.nodes,
            self.graphs.into_iter(),
            self.requirement,
        )
    }
}

impl<BO, B, S, const D: usize, const N: usize> OpsPrep<BO, B, S, D, N, Tracked>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
    BO: Backward<B, D, N, State = S>,
{
    /// Finish the preparation of a tracked operation and returns the output tensor.
    pub fn finish(self, state: S, output: <B as Backend>::TensorPrimitive<D>) -> ADTensor<B, D> {
        let output = ADTensor::from_parents(
            output,
            &self.nodes,
            self.graphs.into_iter(),
            self.requirement,
        );
        let parents = self.nodes.map(|node| node.clone_if_require_grad());
        let ops = Ops::new(parents, output.node.clone(), state);

        output.register_step(OpsStep::new(ops, self.backward))
    }
}

/// Enum used before finishing tracked and untracked operations.
pub enum OpsKind<BO, B, S, const D: usize, const N: usize> {
    Tracked(OpsPrep<BO, B, S, D, N, Tracked>),
    UnTracked(OpsPrep<BO, B, S, D, N, UnTracked>),
}

/// Operation containing its parent nodes, its own node and the backward step state.
#[derive(new, Debug)]
pub struct Ops<S, const N: usize> {
    pub parents: [Option<NodeRef>; N],
    pub node: NodeRef,
    pub state: S,
}

/// Operation implementing backward [step](Step) with type erasing.
#[derive(new, Debug)]
struct OpsStep<B, T, SB, const D: usize, const N: usize>
where
    B: Backend,
    T: Backward<B, D, N, State = SB>,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    ops: Ops<SB, N>,
    backward: T,
    phantom: PhantomData<B>,
}

impl<B, T, SB, const D: usize, const N: usize> Step for OpsStep<B, T, SB, D, N>
where
    B: Backend,
    T: Backward<B, D, N, State = SB>,
    SB: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn step(self: Box<Self>, grads: &mut Gradients) {
        self.backward.backward(self.ops, grads);
    }

    fn node(&self) -> NodeRef {
        self.ops.node.clone()
    }
}
