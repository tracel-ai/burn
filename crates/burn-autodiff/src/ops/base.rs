use super::Backward;
use crate::{
    checkpoint::{
        base::Checkpointer,
        builder::{ActionType, CheckpointerBuilder},
        retro_forward::RetroForward,
    },
    grads::Gradients,
    graph::{ComputingProperty, Graph, NodeID, NodeRef, Requirement, Step},
    tensor::AutodiffTensor,
};
use burn_tensor::{backend::Backend, Shape};
use std::{any::Any, marker::PhantomData, sync::Arc};

#[derive(Debug)]
/// Determines if a node should checkpoint its computed output or its retro_forward for recomputation
/// The action is normally created by the child of the node, once the node is determined to be needed
pub enum CheckpointingAction {
    /// The node's already computed output should be saved
    Computed {
        /// The node
        node_ref: NodeRef,
        /// The node's output
        state_content: Box<dyn Any + Send + Sync>,
    },
    /// The node should recompute itself when asked
    Recompute {
        /// The node
        node_ref: NodeRef,
        /// How the node should recompute itself
        retro_forward: Arc<dyn RetroForward>,
    },
}

impl CheckpointingAction {
    /// Utilitary function to access the id of the node of the checkpointing action
    pub fn id(&self) -> NodeID {
        match self {
            CheckpointingAction::Computed {
                node_ref,
                state_content: _,
            } => node_ref.id.clone(),
            CheckpointingAction::Recompute {
                node_ref,
                retro_forward: _,
            } => node_ref.id.clone(),
        }
    }
}

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
    compute_property: ComputingProperty,
    checkpointer_builder: CheckpointerBuilder,
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

impl<BO, B, S, const D: usize, const N: usize> OpsPrep<BO, B, S, D, N, Init>
where
    B: Backend,
    BO: Backward<B, D, N, State = S>,
{
    /// Indicates that the operation is compute bound, meaning its computation
    /// is heavy and should not be recomputed
    pub fn compute_bound(self) -> OpsPrep<BO, B, S, D, N, ComputePropertyDone> {
        OpsPrep::new(
            self.nodes,
            self.graphs,
            self.requirement,
            self.backward,
            ComputingProperty::ComputeBound,
            self.checkpointer_builder,
        )
    }

    /// Indicates that the operation is memory bound, meaning its computation
    /// is light and can be recomputed
    ///
    pub fn memory_bound(self) -> OpsPrep<BO, B, S, D, N, MemoryBound> {
        OpsPrep::new(
            self.nodes,
            self.graphs,
            self.requirement,
            self.backward,
            self.compute_property,
            self.checkpointer_builder,
        )
    }
}

impl<BO, B, S, const D: usize, const N: usize> OpsPrep<BO, B, S, D, N, MemoryBound>
where
    B: Backend,
    BO: Backward<B, D, N, State = S>,
{
    /// When memory bound, an operation needs to save its RetroForward
    pub fn retro_forward<R: RetroForward>(
        self,
        retro_forward: R,
    ) -> OpsPrep<BO, B, S, D, N, MemoryBoundRetroForward> {
        OpsPrep::new(
            self.nodes,
            self.graphs,
            self.requirement,
            self.backward,
            ComputingProperty::MemoryBound {
                retro_forward: Arc::new(retro_forward),
            },
            self.checkpointer_builder,
        )
    }
}

impl<BO, B, S, const D: usize, const N: usize> OpsPrep<BO, B, S, D, N, MemoryBoundRetroForward>
where
    B: Backend,
    BO: Backward<B, D, N, State = S>,
{
    /// Since the operation may not checkpoint its parents but may need them indirectly
    /// if asked to recompute itself, the method needs to know the parent tensors to maybe checkpoint them
    pub fn parents<'a, B2: Backend, const D2: usize, A: IntoIterator<Item = &'a AutodiffTensor<B2, D2>>>(
        mut self,
        parents: A,
    ) -> OpsPrep<BO, B, S, D, N, ComputePropertyDone> {
        for tensor in parents.into_iter() {
            self.checkpointer_builder
                .checkpoint(tensor, ActionType::Backup);
        }

        OpsPrep::new(
            self.nodes,
            self.graphs,
            self.requirement,
            self.backward,
            self.compute_property,
            self.checkpointer_builder,
        )
    }
}
impl<BO, B, const D: usize, const N: usize> OpsPrep<BO, B, (), D, N, ComputePropertyDone>
where
    B: Backend,
    BO: Backward<B, D, N, State = ()>,
{
    /// Prepare a stateless operation.
    pub fn stateless(
        self,
        output: <B as Backend>::FloatTensorPrimitive<D>,
    ) -> AutodiffTensor<B, D> {
        match self.stateful() {
            OpsKind::Tracked(prep) => prep.finish((), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

impl<BO, B, S, const D: usize, const N: usize> OpsPrep<BO, B, S, D, N, ComputePropertyDone>
where
    B: Backend,
    S: Clone + Send + Sync + std::fmt::Debug + 'static,
    BO: Backward<B, D, N, State = S>,
{
    /// Prepare an operation that requires a state during the backward pass.
    pub fn stateful(self) -> OpsKind<BO, B, S, D, N> {
        match self.requirement.is_none() {
            false => OpsKind::Tracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
                self.compute_property,
                self.checkpointer_builder,
            )),
            true => OpsKind::UnTracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
                self.compute_property,
                self.checkpointer_builder,
            )),
        }
    }
}

/// Duplicated for Init because we can choose to skip compute property chosen (defaults to ambiguous)
impl<BO, B, const D: usize, const N: usize> OpsPrep<BO, B, (), D, N, Init>
where
    B: Backend,
    BO: Backward<B, D, N, State = ()>,
{
    /// Prepare a stateless operation.
    pub fn stateless(
        self,
        output: <B as Backend>::FloatTensorPrimitive<D>,
    ) -> AutodiffTensor<B, D> {
        match self.stateful() {
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
    pub fn stateful(self) -> OpsKind<BO, B, S, D, N> {
        match self.requirement.is_none() {
            false => OpsKind::Tracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
                self.compute_property,
                self.checkpointer_builder,
            )),
            true => OpsKind::UnTracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
                self.compute_property,
                self.checkpointer_builder,
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
    pub fn finish(self, output: <B as Backend>::FloatTensorPrimitive<D>) -> AutodiffTensor<B, D> {
        AutodiffTensor::from_parents(
            output,
            &self.nodes,
            self.graphs.into_iter(),
            self.requirement,
            self.compute_property,
            self.checkpointer_builder,
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
    pub fn finish(
        self,
        state: S,
        output: <B as Backend>::FloatTensorPrimitive<D>,
    ) -> AutodiffTensor<B, D> {
        let output = AutodiffTensor::from_parents(
            output,
            &self.nodes,
            self.graphs.into_iter(),
            self.requirement,
            self.compute_property,
            self.checkpointer_builder,
        );
        let parents = self.nodes.map(|node| node.clone_if_require_grad());
        let ops = Ops::new(parents, output.node.clone(), state);

        output.register_step(OpsStep::new(ops, self.backward))
    }

    /// Checkpoints the tensor
    pub fn checkpoint<const D2: usize>(&mut self, tensor: &AutodiffTensor<B, D2>) -> NodeID {
        self.checkpointer_builder
            .checkpoint(tensor, ActionType::Main);

        tensor.node.id.clone()
    }
}

/// Enum used before finishing tracked and untracked operations.
pub enum OpsKind<BO, B, S, const D: usize, const N: usize> {
    /// Tracked operation preparation.
    Tracked(OpsPrep<BO, B, S, D, N, Tracked>),
    /// Untracked operation preparation.
    UnTracked(OpsPrep<BO, B, S, D, N, UnTracked>),
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
    fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
        self.backward.backward(self.ops, grads, checkpointer);
    }

    fn node(&self) -> NodeRef {
        self.ops.node.clone()
    }
}

/// Make sure the grad tensor has the given shape.
///
/// If broadcasting happened during the forward pass, the gradients will be sum along the
/// broadcasted dimension.
pub fn broadcast_shape<B: Backend, const D: usize>(
    mut grad: B::FloatTensorPrimitive<D>,
    shape: &Shape<D>,
) -> B::FloatTensorPrimitive<D> {
    let shape_grad = B::float_shape(&grad);

    for i in 0..D {
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
