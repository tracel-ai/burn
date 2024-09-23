use super::Backward;
use crate::{
    checkpoint::{
        base::Checkpointer,
        builder::{ActionType, CheckpointerBuilder},
        retro_forward::RetroForward,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    graph::{ComputingProperty, NodeID, NodeRef, Requirement, Step},
    tensor::AutodiffTensor,
};
use burn_tensor::{backend::Backend, Shape};
use std::marker::PhantomData;

/// Operation in preparation.
///
/// Each mode has its own set of functions to minimize cloning for unused backward states.
#[derive(new)]
pub struct OpsPrep<Backward, B, S, C, const N: usize, Mode = Init> {
    nodes: [NodeRef; N],
    requirement: Requirement,
    backward: Backward,
    compute_property: ComputingProperty,
    checkpointer_builder: CheckpointerBuilder,
    checkpoint_strategy: PhantomData<C>,
    phantom_backend: PhantomData<B>,
    phantom_state: PhantomData<S>,
    marker: PhantomData<Mode>,
}

/// Operation is initialized
pub struct Init;
/// Operation has been tagged as memory bound
pub struct MemoryBound;
/// Memory bound operation has received its RetroForward
pub struct MemoryBoundRetroForward;
/// Operation's compute property is fixed
pub struct ComputePropertyDone;
/// Tracked operation tag.
pub struct Tracked;
/// Untracked operation tag.
pub struct UnTracked;

impl<BO, B, S, C, const N: usize> OpsPrep<BO, B, S, C, N, Init>
where
    B: Backend,
    BO: Backward<B, N, State = S>,
{
    /// Indicates that the operation is compute bound, meaning its computation
    /// is heavy and should not be recomputed
    pub fn compute_bound(self) -> OpsPrep<BO, B, S, C, N, ComputePropertyDone> {
        OpsPrep::new(
            self.nodes,
            self.requirement,
            self.backward,
            ComputingProperty::ComputeBound,
            self.checkpointer_builder,
        )
    }

    /// Indicates that the operation is memory bound, meaning its computation
    /// is light and can be recomputed
    pub fn memory_bound(self) -> OpsPrep<BO, B, S, C, N, MemoryBound> {
        OpsPrep::new(
            self.nodes,
            self.requirement,
            self.backward,
            self.compute_property,
            self.checkpointer_builder,
        )
    }
}

impl<BO, B, S, C, const N: usize> OpsPrep<BO, B, S, C, N, MemoryBound>
where
    B: Backend,
    BO: Backward<B, N, State = S>,
    C: CheckpointStrategy,
{
    /// Registers the retro forward, if needed
    pub fn retro_forward<R: RetroForward>(
        self,
        retro_forward: R,
    ) -> OpsPrep<BO, B, S, C, N, MemoryBoundRetroForward> {
        OpsPrep::new(
            self.nodes,
            self.requirement,
            self.backward,
            C::compute_property(retro_forward),
            self.checkpointer_builder,
        )
    }
}

impl<BO, B, S, C, const N: usize> OpsPrep<BO, B, S, C, N, MemoryBoundRetroForward>
where
    B: Backend,
    BO: Backward<B, N, State = S>,
    C: CheckpointStrategy,
{
    /// Checkpoints the parents, if needed
    pub fn parents<'a, B2, A>(mut self, parents: A) -> OpsPrep<BO, B, S, C, N, ComputePropertyDone>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>,
    {
        C::checkpoint_parents(parents, &mut self.checkpointer_builder);

        OpsPrep::new(
            self.nodes,
            self.requirement,
            self.backward,
            self.compute_property,
            self.checkpointer_builder,
        )
    }
}

impl<BO, B, C, const N: usize> OpsPrep<BO, B, (), C, N, ComputePropertyDone>
where
    B: Backend,
    BO: Backward<B, N, State = ()>,
{
    /// Prepare a stateless operation.
    pub fn stateless(self, output: <B as Backend>::FloatTensorPrimitive) -> AutodiffTensor<B> {
        match self.stateful() {
            OpsKind::Tracked(prep) => prep.finish((), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

impl<BO, B, S, C, const N: usize> OpsPrep<BO, B, S, C, N, ComputePropertyDone>
where
    B: Backend,
    S: Clone + Send + std::fmt::Debug + 'static,
    BO: Backward<B, N, State = S>,
{
    /// Prepare an operation that requires a state during the backward pass.
    pub fn stateful(self) -> OpsKind<BO, B, S, C, N> {
        match self.requirement.is_none() {
            false => OpsKind::Tracked(OpsPrep::new(
                self.nodes,
                self.requirement,
                self.backward,
                self.compute_property,
                self.checkpointer_builder,
            )),
            true => OpsKind::UnTracked(OpsPrep::new(
                self.nodes,
                self.requirement,
                self.backward,
                self.compute_property,
                self.checkpointer_builder,
            )),
        }
    }
}

impl<BO, B, S, C, const N: usize> OpsPrep<BO, B, S, C, N, UnTracked>
where
    B: Backend,
    S: Clone + Send + std::fmt::Debug + 'static,
    BO: Backward<B, N, State = S>,
{
    /// Finish the preparation of an untracked operation and returns the output tensor.
    pub fn finish(self, output: <B as Backend>::FloatTensorPrimitive) -> AutodiffTensor<B> {
        let output = AutodiffTensor::from_parents(
            output,
            &self.nodes,
            self.requirement,
            self.compute_property,
        );
        let parents = self.nodes.map(|node| node.clone_if_require_grad());
        let ops = Ops::new(parents, output.node.clone(), ());

        // We register the ops in the graph even if untracked, otherwise memory bound operations
        // that have an untracked parent would not be able to retrieve it
        output.register_step(UntrackedOpsStep::new(ops), self.checkpointer_builder)
    }
}

impl<BO, B, S, C, const N: usize> OpsPrep<BO, B, S, C, N, Tracked>
where
    B: Backend,
    S: Clone + Send + std::fmt::Debug + 'static,
    BO: Backward<B, N, State = S>,
{
    /// Finish the preparation of a tracked operation and returns the output tensor.
    pub fn finish(
        self,
        state: S,
        output: <B as Backend>::FloatTensorPrimitive,
    ) -> AutodiffTensor<B> {
        let output = AutodiffTensor::from_parents(
            output,
            &self.nodes,
            self.requirement,
            self.compute_property,
        );
        let parents = self.nodes.map(|node| node.clone_if_require_grad());
        let ops = Ops::new(parents, output.node.clone(), state);

        output.register_step(OpsStep::new(ops, self.backward), self.checkpointer_builder)
    }

    /// Checkpoints the tensor
    pub fn checkpoint(&mut self, tensor: &AutodiffTensor<B>) -> NodeID {
        self.checkpointer_builder
            .checkpoint(tensor, ActionType::Explicit);

        tensor.node.id
    }
}

/// Enum used before finishing tracked and untracked operations.
pub enum OpsKind<BO, B, S, C, const N: usize> {
    /// Tracked operation preparation.
    Tracked(OpsPrep<BO, B, S, C, N, Tracked>),
    /// Untracked operation preparation.
    UnTracked(OpsPrep<BO, B, S, C, N, UnTracked>),
}

/// Operation containing its parent nodes, its own node and the backward step state.
#[derive(new, Debug)]
pub struct Ops<S, const N: usize> {
    /// Parents nodes.
    pub parents: [Option<NodeRef>; N],
    /// The node.
    pub node: NodeRef,
    /// The state.
    pub state: S,
}

/// Operation implementing backward [step](Step) with type erasing.
#[derive(new, Debug)]
struct OpsStep<B, T, SB, const N: usize>
where
    B: Backend,
    T: Backward<B, N, State = SB>,
    SB: Clone + Send + std::fmt::Debug + 'static,
{
    ops: Ops<SB, N>,
    backward: T,
    phantom: PhantomData<B>,
}

impl<B, T, SB, const N: usize> Step for OpsStep<B, T, SB, N>
where
    B: Backend,
    T: Backward<B, N, State = SB>,
    SB: Clone + Send + std::fmt::Debug + 'static,
{
    fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
        self.backward.backward(self.ops, grads, checkpointer);
    }

    fn node(&self) -> NodeID {
        self.ops.node.id
    }

    fn parents(&self) -> Vec<NodeID> {
        self.ops.node.parents.clone()
    }

    fn depth(&self) -> usize {
        self.ops.node.order
    }
}

#[derive(new, Debug)]
struct UntrackedOpsStep<const N: usize> {
    ops: Ops<(), N>,
}

impl<const N: usize> Step for UntrackedOpsStep<N> {
    fn step(self: Box<Self>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        // Nothing to do
    }

    fn node(&self) -> NodeID {
        self.ops.node.id
    }

    fn parents(&self) -> Vec<NodeID> {
        self.ops.node.parents.clone()
    }
    fn depth(&self) -> usize {
        self.ops.node.order
    }
}

/// Make sure the grad tensor has the given shape.
///
/// If broadcasting happened during the forward pass, the gradients will be sum along the
/// broadcasted dimension.
pub fn broadcast_shape<B: Backend>(
    mut grad: B::FloatTensorPrimitive,
    shape: &Shape,
) -> B::FloatTensorPrimitive {
    let shape_grad = B::float_shape(&grad);
    let ndims = shape_grad.num_dims();

    for i in 0..ndims {
        if shape_grad.dims[i] != shape.dims[i] {
            if shape.dims[i] != 1 {
                panic!(
                    "Invalid broadcast shapes: Next grad shape {:?}, Previous grad shape {:?}. {}",
                    shape.dims, shape_grad.dims, "Expected the shape of the next grad to be 1."
                );
            }
            grad = B::float_sum_dim(grad, i);
        }
    }

    grad
}
