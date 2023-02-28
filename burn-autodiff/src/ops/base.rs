use std::marker::PhantomData;

use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::{
        NodeRef, Requirement, {Graph, Step},
    },
    tensor::ADTensor,
    utils::duplicate,
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
    fn backward(self, ops: Ops<Self::State, N>, grads: &mut Gradients);

    fn prepare(
        self,
        nodes: [NodeRef; N],
        graphs: [Graph; N],
    ) -> PrepareStep<Self, B, Self::State, D, N> {
        let requirement = Requirement::from_nodes(&nodes);
        PrepareStep::new(nodes, graphs, requirement, self)
    }
}

type TensorPrimitive<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

pub struct Init;
pub struct Tracked;
pub struct Untracked;

#[derive(new)]
pub struct PrepareStep<Backward, B, S, const D: usize, const N: usize, Mode = Init> {
    nodes: [NodeRef; N],
    graphs: [Graph; N],
    requirement: Requirement,
    backward: Backward,
    phantom_backend: PhantomData<B>,
    phantom_state: PhantomData<S>,
    marker: PhantomData<Mode>,
}

impl<BO, B, S, const D: usize, const N: usize> PrepareStep<BO, B, S, D, N, Init>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
    BO: Backward<B, D, N, State = S>,
{
    pub fn statefull(self) -> PrepKind<BO, B, S, D, N> {
        match self.requirement.is_none() {
            false => PrepKind::Tracked(PrepareStep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
            )),
            true => PrepKind::Untracked(PrepareStep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
            )),
        }
    }
}

impl<BO, B, S, const D: usize, const N: usize> PrepareStep<BO, B, S, D, N, Untracked>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
    BO: Backward<B, D, N, State = S>,
{
    pub fn finish(self, output: TensorPrimitive<B, D>) -> ADTensor<B, D> {
        ADTensor::from_ops(
            &self.nodes,
            output,
            self.graphs.into_iter(),
            self.requirement,
        )
    }
}

impl<BO, B, S, const D: usize, const N: usize> PrepareStep<BO, B, S, D, N, Tracked>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
    BO: Backward<B, D, N, State = S>,
{
    pub fn finish(self, state: S, output: TensorPrimitive<B, D>) -> ADTensor<B, D> {
        let output = ADTensor::from_ops(
            &self.nodes,
            output,
            self.graphs.into_iter(),
            self.requirement,
        );
        let parents = self.nodes.map(|node| node.clone_if_require_grad());
        let ops = Ops::new(parents, output.node.clone(), state);

        output.register_step(OpsStep::new(ops, self.backward))
    }
}

pub enum PrepKind<BO, B, S, const D: usize, const N: usize> {
    Tracked(PrepareStep<BO, B, S, D, N, Tracked>),
    Untracked(PrepareStep<BO, B, S, D, N, Untracked>),
}

impl<BO, B, const D: usize, const N: usize> PrepareStep<BO, B, (), D, N, Init>
where
    B: Backend,
    BO: Backward<B, D, N, State = ()>,
{
    pub fn stateless(self, output: TensorPrimitive<B, D>) -> ADTensor<B, D> {
        match self.statefull() {
            PrepKind::Tracked(prep) => prep.finish((), output),
            PrepKind::Untracked(prep) => prep.finish(output),
        }
    }
}

#[derive(new, Debug)]
pub struct Ops<S, const N: usize> {
    pub parents: OpsNodes<N>,
    pub node: NodeRef,
    pub state: S,
}

pub type OpsNodes<const N: usize> = [Option<NodeRef>; N];

pub fn binary<B, const D_OUT: usize, const D_LHS: usize, const D_RHS: usize, FLhs, FRhs>(
    parents: OpsNodes<2>,
    node: NodeRef,
    grads: &mut Gradients,
    func_lhs: FLhs,
    func_rhs: FRhs,
) where
    B: Backend,
    FLhs: FnOnce(B::TensorPrimitive<D_OUT>) -> B::TensorPrimitive<D_LHS>,
    FRhs: FnOnce(B::TensorPrimitive<D_OUT>) -> B::TensorPrimitive<D_RHS>,
{
    let [grad_4lhs, grad_4rhs] = duplicate(&parents, Some(grads.consume::<B, D_OUT>(&node)));
    let [node_lhs, node_rhs] = parents;

    node_lhs.map(|node| {
        let grad = func_lhs(grad_4lhs.unwrap());
        grads.register::<B, D_LHS>(node, grad)
    });

    node_rhs.map(|node| {
        let grad = func_rhs(grad_4rhs.unwrap());
        grads.register::<B, D_RHS>(node, grad)
    });
}

pub fn unary<B, const D_OUT: usize, const D_IN: usize, F>(
    parents: OpsNodes<1>,
    node: NodeRef,
    grads: &mut Gradients,
    func: F,
) where
    B: Backend,
    F: FnOnce(B::TensorPrimitive<D_OUT>) -> B::TensorPrimitive<D_IN>,
{
    let [parent_node] = parents;
    let grad = grads.consume::<B, D_OUT>(&node);

    parent_node.map(|node| {
        let grad = func(grad);
        grads.register::<B, D_IN>(node, grad)
    });
}

pub fn unary_different_backend<BIn, BOut, const D_OUT: usize, const D_IN: usize, F>(
    parents: OpsNodes<1>,
    node: NodeRef,
    grads: &mut Gradients,
    func: F,
) where
    BIn: Backend,
    BOut: Backend,
    F: FnOnce(BOut::TensorPrimitive<D_OUT>) -> BIn::TensorPrimitive<D_IN>,
{
    let [parent_node] = parents;
    let grad = grads.consume::<BOut, D_OUT>(&node);

    parent_node.map(|node| {
        let grad = func(grad);
        grads.register::<BIn, D_IN>(node, grad)
    });
}

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
