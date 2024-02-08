use super::Backward;
use crate::{
    checkpoint::base::{Checkpointer, RetroForward},
    grads::Gradients,
    graph::{ComputingProperties, Graph, NodeID, NodeRef, Requirement, Step},
    tensor::AutodiffTensor,
};
use burn_tensor::{backend::Backend, Shape};
use std::{marker::PhantomData, sync::Arc};

pub enum CheckpointStrategy {
    Computed,
    Recompute {
        retro_forward: Box<dyn RetroForward>,
    },
}

pub enum StateStrategy {
    Saved,
    FromInputs,
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
    compute_properties: ComputingProperties,
    checkpointer: Option<Checkpointer>,
    phantom_backend: PhantomData<B>,
    phantom_state: PhantomData<S>,
    marker: PhantomData<Mode>,
}

/// Init operation tag.
pub struct Init;
/// Tracked operation tag.
pub struct Tracked;
/// Untracked operation tag.
pub struct UnTracked;

impl<BO, B, S, const D: usize, const N: usize> OpsPrep<BO, B, S, D, N, Init>
where
    B: Backend,
    BO: Backward<B, D, N, State = S>,
{
    pub fn compute_bound(self) -> OpsPrep<BO, B, S, D, N, Init> {
        OpsPrep::new(
            self.nodes,
            self.graphs,
            self.requirement,
            self.backward,
            ComputingProperties::ComputeBound,
            self.checkpointer,
        )
    }
    pub fn memory_bound<R: RetroForward>(self, retro_forward: R) -> OpsPrep<BO, B, S, D, N, Init> {
        OpsPrep::new(
            self.nodes,
            self.graphs,
            self.requirement,
            self.backward,
            ComputingProperties::MemoryBound {
                retro_forward: Arc::new(retro_forward),
            },
            self.checkpointer,
        )
    }
}

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
                self.compute_properties,
                self.checkpointer,
            )),
            true => OpsKind::UnTracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.backward,
                self.compute_properties,
                self.checkpointer,
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
            self.compute_properties,
            self.checkpointer,
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
            self.compute_properties,
            self.checkpointer,
        );
        let parents = self.nodes.map(|node| node.clone_if_require_grad());
        let ops = Ops::new(parents, output.node.clone(), state);

        output.register_step(OpsStep::new(ops, self.backward))
    }

    pub fn checkpoint<const D2: usize>(&mut self, tensor: &AutodiffTensor<B, D2>) -> NodeID {
        if self.checkpointer.is_none() {
            self.checkpointer = Some(Checkpointer::default())
        }
        match &tensor.node.properties {
            ComputingProperties::ComputeBound => self
                .checkpointer
                .as_mut()
                .unwrap()
                .checkpoint_compute(tensor.node.clone(), tensor.primitive.clone()),
            ComputingProperties::MemoryBound { retro_forward } => self
                .checkpointer
                .as_mut()
                .unwrap()
                .checkpoint_lazy(tensor.node.clone(), retro_forward.clone()),
            // TODO Ambiguous are considered as compute bound for now so we don't need to provide retro_forward
            ComputingProperties::Ambiguous => self
                .checkpointer
                .as_mut()
                .unwrap()
                .checkpoint_compute(tensor.node.clone(), tensor.primitive.clone()),
        }
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
