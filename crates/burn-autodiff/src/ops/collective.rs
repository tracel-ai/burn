use alloc::{boxed::Box, vec::Vec};
use burn_backend::{
    Backend, DType, TensorMetadata,
    distributed::ReduceOperation,
    ops::{CollectiveOps, DimSplit, EmulatedCollectives},
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Shape, sync::Arc};

use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder, strategy::CheckpointStrategy},
    grads::Gradients,
    graph::{ComputingProperty, Node, NodeRef, Parent, Requirement, Step},
    runtime::AutodiffClient,
    tensor::AutodiffTensor,
};

// Every collective runs on the inner backend, the float ones recorded with a backward of their own.
// Broadcast keeps its default: no backend has a native one, and the transfers it is made of already
// sum the gradients back onto the root.
impl<B: Backend, C: CheckpointStrategy> CollectiveOps<Self> for Autodiff<B, C> {
    fn float_all_reduce(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        FloatCollective::AllReduce { op }.run(tensors)
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        FloatCollective::AllGather { dim }.run(tensors)
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        FloatCollective::ReduceScatter { dim, op }.run(tensors)
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        FloatCollective::AllToAll {
            split_dim,
            concat_dim,
        }
        .run(tensors)
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        B::int_all_reduce(tensors, op)
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        B::int_all_gather(tensors, dim)
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        B::int_reduce_scatter(tensors, dim, op)
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        B::int_all_to_all(tensors, split_dim, concat_dim)
    }

    fn int_broadcast(tensor: IntTensor<Self>, devices: &[Device<Self>]) -> Vec<IntTensor<Self>> {
        B::int_broadcast(tensor, devices)
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        B::bool_all_gather(tensors, dim)
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        B::bool_all_to_all(tensors, split_dim, concat_dim)
    }

    fn bool_broadcast(tensor: BoolTensor<Self>, devices: &[Device<Self>]) -> Vec<BoolTensor<Self>> {
        B::bool_broadcast(tensor, devices)
    }
}

/// A float collective over every member, whose backward is another collective over them.
#[derive(Clone, Copy, Debug)]
enum FloatCollective {
    /// Every member receives `op` over every member's tensor.
    AllReduce { op: ReduceOperation },
    /// Every member receives every member's tensor, concatenated along `dim`.
    AllGather { dim: usize },
    /// Member `i` receives chunk `i` along `dim` of `op` over every member's tensor.
    ReduceScatter { dim: usize, op: ReduceOperation },
    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated
    /// along `concat_dim`.
    AllToAll { split_dim: usize, concat_dim: usize },
}

impl FloatCollective {
    /// Runs the collective on the inner backend and records its backward when any member is
    /// tracked.
    fn run<B: Backend>(self, tensors: Vec<AutodiffTensor<B>>) -> Vec<AutodiffTensor<B>> {
        let (primitives, guards): (Vec<_>, Vec<_>) =
            tensors.into_iter().map(AutodiffTensor::into_parts).unzip();
        let inputs: Vec<NodeRef> = guards
            .iter()
            .map(|guard| guard.node_ref().clone())
            .collect();
        let outputs = match Requirement::from_nodes(&inputs) {
            Requirement::None => self.untracked(primitives, &inputs),
            requirement => self.tracked(primitives, &inputs, requirement),
        };
        drop(guards);
        outputs
    }

    /// The outputs with no backward recorded, since no member needs a gradient.
    fn untracked<B: Backend>(
        self,
        primitives: Vec<FloatTensor<B>>,
        inputs: &[NodeRef],
    ) -> Vec<AutodiffTensor<B>> {
        self.forward::<B>(primitives)
            .into_iter()
            .map(|output| {
                AutodiffTensor::from_parents(
                    output,
                    inputs,
                    Requirement::None,
                    ComputingProperty::ComputeBound,
                )
            })
            .collect()
    }

    /// The outputs as children of the collective's own node, whose backward takes all their
    /// gradients at once.
    fn tracked<B: Backend>(
        self,
        primitives: Vec<FloatTensor<B>>,
        inputs: &[NodeRef],
        requirement: Requirement,
    ) -> Vec<AutodiffTensor<B>> {
        let input_shape = primitives[0].shape();
        let node: NodeRef =
            Node::from_parents(inputs, requirement, ComputingProperty::ComputeBound).into();
        let outputs: Vec<AutodiffTensor<B>> = self
            .forward::<B>(primitives)
            .into_iter()
            .map(|output| {
                AutodiffTensor::from_parents(
                    output,
                    core::slice::from_ref(&node),
                    requirement,
                    ComputingProperty::ComputeBound,
                )
            })
            .collect();

        let backward = CollectiveBackward {
            forward: self,
            inputs: inputs.iter().map(Node::clone_if_require_grad).collect(),
            input_shape,
            outputs: outputs.iter().map(CollectiveOutput::new).collect(),
            node: node.clone(),
        };
        // Registered before the outputs' steps, so that they join its graph.
        node.client.register(
            Arc::new(node.id),
            Box::new(backward),
            CheckpointerBuilder::default(),
        );
        outputs
            .into_iter()
            .map(|output| {
                let step = CollectiveOutputStep {
                    node: output.node.clone(),
                };
                output.register_step(step, CheckpointerBuilder::default())
            })
            .collect()
    }

    /// The collective itself, on the inner backend.
    fn forward<B: Backend>(self, tensors: Vec<FloatTensor<B>>) -> Vec<FloatTensor<B>> {
        match self {
            Self::AllReduce { op } => B::float_all_reduce(tensors, op),
            Self::AllGather { dim } => B::float_all_gather(tensors, dim),
            Self::ReduceScatter { dim, op } => B::float_reduce_scatter(tensors, dim, op),
            Self::AllToAll {
                split_dim,
                concat_dim,
            } => B::float_all_to_all(tensors, split_dim, concat_dim),
        }
    }

    /// The members' input gradients from their output gradients: all-reduce is its own adjoint,
    /// all-gather and reduce-scatter are each other's, and all-to-all swaps its dims.
    fn backward<B: Backend>(self, grads: Vec<FloatTensor<B>>) -> Vec<FloatTensor<B>> {
        match self {
            Self::AllReduce { op } => B::float_all_reduce(grads, op),
            Self::AllGather { dim } => B::float_reduce_scatter(grads, dim, ReduceOperation::Sum),
            Self::ReduceScatter { dim, op } => Self::reduce_scatter_backward::<B>(grads, dim, op),
            Self::AllToAll {
                split_dim,
                concat_dim,
            } => Self::all_to_all_backward::<B>(grads, split_dim, concat_dim),
        }
    }

    /// Every member's output gradient gathered onto each member, divided by the member count for a
    /// mean.
    fn reduce_scatter_backward<B: Backend>(
        grads: Vec<FloatTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<B>> {
        let members = grads.len();
        let grads: Vec<_> = grads
            .into_iter()
            .map(|grad| match op {
                ReduceOperation::Sum => grad,
                ReduceOperation::Mean => B::float_div_scalar(grad, members.into()),
            })
            .collect();
        match Self::alike(&grads) {
            true => B::float_all_gather(grads, dim),
            // Uneven chunks, which a collective refuses and the emulation concatenates as they are.
            false => EmulatedCollectives::<B>::float_all_gather(grads, dim),
        }
    }

    /// The all-to-all with its dims swapped, sending each chunk's gradient back to its member.
    fn all_to_all_backward<B: Backend>(
        grads: Vec<FloatTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<B>> {
        match Self::alike(&grads) {
            true => B::float_all_to_all(grads, concat_dim, split_dim),
            false => Self::uneven_all_to_all_backward::<B>(grads, split_dim, concat_dim),
        }
    }

    /// The all-to-all's backward when uneven chunks left the gradients of different shapes, which
    /// no collective takes.
    fn uneven_all_to_all_backward<B: Backend>(
        grads: Vec<FloatTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<B>> {
        (0..grads.len())
            .map(|member| Self::uneven_member_gradient::<B>(&grads, member, split_dim, concat_dim))
            .collect()
    }

    /// Piece `member` along `concat_dim` of every output gradient, each cut into equal pieces of
    /// its own length, concatenated along `split_dim` on the member's device.
    fn uneven_member_gradient<B: Backend>(
        grads: &[FloatTensor<B>],
        member: usize,
        split_dim: usize,
        concat_dim: usize,
    ) -> FloatTensor<B> {
        let device = grads[member].device();
        let pieces = grads
            .iter()
            .map(|grad| {
                let split = DimSplit {
                    dim: concat_dim,
                    len: grad.shape()[concat_dim],
                    members: grads.len(),
                };
                let piece = B::float_slice(grad.clone(), &split.slices(member));
                B::float_to_device(piece, &device)
            })
            .collect();
        B::float_cat(pieces, split_dim)
    }

    /// Whether every tensor has one shape, which a native collective requires of its members.
    fn alike<T: TensorMetadata>(grads: &[T]) -> bool {
        grads
            .windows(2)
            .all(|pair| pair[0].shape() == pair[1].shape())
    }
}

/// The backward of one collective, which needs every member's output gradient at once.
///
/// Each output's own step leaves its gradient registered for this one, which runs after them all
/// since every output is a child of its node.
#[derive(Debug)]
struct CollectiveBackward<B: Backend> {
    /// The collective run forward, whose backward this step runs.
    forward: FloatCollective,
    /// The collective's own node: a child of every input, the parent of every output.
    node: NodeRef,
    /// Each member's input node, `None` where that member needs no gradient.
    inputs: Vec<Option<NodeRef>>,
    /// The shape every member's input shares.
    input_shape: Shape,
    /// Each member's output.
    outputs: Vec<CollectiveOutput<B>>,
}

impl<B: Backend> Step for CollectiveBackward<B> {
    /// Takes every output's gradient, with zeros for one the loss never used, runs the backward
    /// collective, and registers each tracked input's gradient.
    fn step(self: Box<Self>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let output_grads: Vec<_> = self
            .outputs
            .iter()
            .map(|output| grads.consume_if_registered::<B>(&output.node))
            .collect();
        if output_grads.iter().all(Option::is_none) {
            return;
        }

        let input_grads = match self.input_shape.num_elements() {
            // The backward would split a dimension that holds no element, which a collective refuses.
            0 => self
                .outputs
                .iter()
                .map(|output| output.zeros(self.input_shape.clone()))
                .collect(),
            _ => {
                let output_grads = self
                    .outputs
                    .iter()
                    .zip(output_grads)
                    .map(|(output, grad)| {
                        grad.unwrap_or_else(|| output.zeros(output.shape.clone()))
                    })
                    .collect();
                self.forward.backward::<B>(output_grads)
            }
        };

        for (input, grad) in self.inputs.iter().zip(input_grads) {
            if let Some(input) = input {
                grads.register::<B>(input.id, grad);
            }
        }
    }

    /// The collective's node's depth, one past its deepest input.
    fn depth(&self) -> usize {
        self.node.order
    }

    /// The tracked inputs.
    fn parents(&self) -> &[Parent] {
        &self.node.parents
    }
}

/// A member's output, with what it takes to stand in a zero gradient when the loss never used it.
#[derive(Debug)]
struct CollectiveOutput<B: Backend> {
    /// The output's node, under which its gradient is registered.
    node: NodeRef,
    /// The output's shape, which differs between members when chunks are uneven.
    shape: Shape,
    /// The member's device.
    device: Device<B>,
    /// The output's dtype, which its gradient shares.
    dtype: DType,
}

impl<B: Backend> CollectiveOutput<B> {
    /// What the backward needs of `output`, without holding its tensor.
    fn new(output: &AutodiffTensor<B>) -> Self {
        Self {
            node: output.node.clone(),
            shape: output.primitive.shape(),
            device: output.primitive.device(),
            dtype: output.primitive.dtype(),
        }
    }

    /// Zeros of `shape` on the member's device, in the output's dtype.
    fn zeros(&self, shape: Shape) -> FloatTensor<B> {
        B::float_zeros(shape, &self.device, self.dtype.into())
    }
}

/// An output's step, which does nothing: it only places the output on the graph, as a child of
/// the collective's node, and leaves the output's gradient for that node's backward.
#[derive(Debug)]
struct CollectiveOutputStep {
    /// The output's node.
    node: NodeRef,
}

impl Step for CollectiveOutputStep {
    /// Leaves the output's gradient registered, for the collective's backward to consume.
    fn step(self: Box<Self>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {}

    /// One past the collective's node, so this step runs before that node's backward.
    fn depth(&self) -> usize {
        self.node.order
    }

    /// The collective's node.
    fn parents(&self) -> &[Parent] {
        &self.node.parents
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{TensorData, ops::FloatTensorOps};
    use burn_flex::{Flex, FlexDevice};

    type TestAutodiff = Autodiff<Flex>;

    /// The emulation would record transfers, slices and concatenations between the members and
    /// every output, each output a child of several of them.
    #[test]
    fn a_float_collective_records_one_step_over_its_members() {
        let members = || -> Vec<_> {
            (0..3)
                .map(|_| {
                    let data = TensorData::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
                    AutodiffTensor::new(Flex::float_from_data(data, &FlexDevice)).require_grad()
                })
                .collect()
        };
        let collectives = [
            TestAutodiff::float_all_reduce(members(), ReduceOperation::Sum),
            TestAutodiff::float_all_gather(members(), 0),
            TestAutodiff::float_reduce_scatter(members(), 0, ReduceOperation::Sum),
            TestAutodiff::float_all_to_all(members(), 0, 1),
        ];

        for outputs in collectives {
            let step = &outputs[0].node.parents;
            assert_eq!(step.len(), 1);
            for output in &outputs {
                assert_eq!(&output.node.parents, step);
                assert_eq!(output.node.order, 2);
            }
        }
    }
}
