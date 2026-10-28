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
        FloatCollective::AllReduce { op }.record(tensors)
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        FloatCollective::AllGather { dim }.record(tensors)
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        FloatCollective::ReduceScatter { dim, op }.record(tensors)
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
        .record(tensors)
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
    AllReduce { op: ReduceOperation },
    AllGather { dim: usize },
    ReduceScatter { dim: usize, op: ReduceOperation },
    AllToAll { split_dim: usize, concat_dim: usize },
}

/// The backward of one collective, which needs every member's output gradient at once.
///
/// Each output's own step leaves its gradient registered for this one, which runs after them all
/// since every output is a child of its node.
#[derive(Debug)]
struct CollectiveStep<B: Backend> {
    collective: FloatCollective,
    node: NodeRef,
    inputs: Vec<Option<NodeRef>>,
    input_shape: Shape,
    outputs: Vec<CollectiveOutput<B>>,
}

/// A member's output, with what it takes to stand in a zero gradient when the loss never used it.
#[derive(Debug)]
struct CollectiveOutput<B: Backend> {
    node: NodeRef,
    shape: Shape,
    device: Device<B>,
    dtype: DType,
}

#[derive(Debug)]
struct CollectiveOutputStep {
    node: NodeRef,
}

impl FloatCollective {
    /// Runs the collective on the inner backend, its outputs the children of one backward step.
    fn record<B: Backend>(self, tensors: Vec<AutodiffTensor<B>>) -> Vec<AutodiffTensor<B>> {
        let (primitives, guards): (Vec<_>, Vec<_>) =
            tensors.into_iter().map(AutodiffTensor::into_parts).unzip();
        let inputs: Vec<NodeRef> = guards
            .iter()
            .map(|guard| guard.node_ref().clone())
            .collect();
        let requirement = Requirement::from_nodes(&inputs);
        let property = ComputingProperty::ComputeBound;

        if requirement.is_none() {
            return self
                .forward::<B>(primitives)
                .into_iter()
                .map(|output| {
                    AutodiffTensor::from_parents(output, &inputs, requirement, property.clone())
                })
                .collect();
        }

        let input_shape = primitives[0].shape();
        let node: NodeRef = Node::from_parents(&inputs, requirement, property.clone()).into();
        let outputs: Vec<AutodiffTensor<B>> = self
            .forward::<B>(primitives)
            .into_iter()
            .map(|output| {
                AutodiffTensor::from_parents(
                    output,
                    core::slice::from_ref(&node),
                    requirement,
                    property.clone(),
                )
            })
            .collect();

        let step = CollectiveStep {
            collective: self,
            node: node.clone(),
            inputs: inputs.iter().map(Node::clone_if_require_grad).collect(),
            input_shape,
            outputs: outputs.iter().map(CollectiveOutput::new).collect(),
        };
        // Registered before its outputs, so that they join its graph.
        node.client.register(
            Arc::new(node.id),
            Box::new(step),
            CheckpointerBuilder::default(),
        );
        let outputs = outputs
            .into_iter()
            .map(|output| {
                let step = CollectiveOutputStep {
                    node: output.node.clone(),
                };
                output.register_step(step, CheckpointerBuilder::default())
            })
            .collect();
        drop(guards);
        outputs
    }

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

    /// The members' input gradients from their output gradients.
    fn adjoint<B: Backend>(self, grads: Vec<FloatTensor<B>>) -> Vec<FloatTensor<B>> {
        match self {
            Self::AllReduce { op } => B::float_all_reduce(grads, op),
            Self::AllGather { dim } => B::float_reduce_scatter(grads, dim, ReduceOperation::Sum),
            Self::ReduceScatter { dim, op } => Self::reduce_scatter_adjoint::<B>(grads, dim, op),
            Self::AllToAll {
                split_dim,
                concat_dim,
            } => Self::all_to_all_adjoint::<B>(grads, split_dim, concat_dim),
        }
    }

    fn reduce_scatter_adjoint<B: Backend>(
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

    fn all_to_all_adjoint<B: Backend>(
        grads: Vec<FloatTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<B>> {
        match Self::alike(&grads) {
            true => B::float_all_to_all(grads, concat_dim, split_dim),
            false => Self::uneven_all_to_all_adjoint::<B>(grads, split_dim, concat_dim),
        }
    }

    /// Member `j` takes piece `j` along `concat_dim` of every output gradient, each cut into equal
    /// pieces of its own length, since uneven chunks left the gradients of different shapes.
    fn uneven_all_to_all_adjoint<B: Backend>(
        grads: Vec<FloatTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<B>> {
        let members = grads.len();
        grads
            .iter()
            .enumerate()
            .map(|(member, own)| {
                let device = own.device();
                let pieces = grads
                    .iter()
                    .map(|grad| {
                        let split = DimSplit {
                            dim: concat_dim,
                            len: grad.shape()[concat_dim],
                            members,
                        };
                        let piece = B::float_slice(grad.clone(), &split.slices(member));
                        B::float_to_device(piece, &device)
                    })
                    .collect();
                B::float_cat(pieces, split_dim)
            })
            .collect()
    }

    fn alike<T: TensorMetadata>(grads: &[T]) -> bool {
        grads
            .windows(2)
            .all(|pair| pair[0].shape() == pair[1].shape())
    }
}

impl<B: Backend> Step for CollectiveStep<B> {
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
            // An adjoint would split a dimension that holds no element, which a collective refuses.
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
                self.collective.adjoint::<B>(output_grads)
            }
        };

        for (input, grad) in self.inputs.iter().zip(input_grads) {
            if let Some(input) = input {
                grads.register::<B>(input.id, grad);
            }
        }
    }

    fn depth(&self) -> usize {
        self.node.order
    }

    fn parents(&self) -> &[Parent] {
        &self.node.parents
    }
}

impl<B: Backend> CollectiveOutput<B> {
    fn new(output: &AutodiffTensor<B>) -> Self {
        Self {
            node: output.node.clone(),
            shape: output.primitive.shape(),
            device: output.primitive.device(),
            dtype: output.primitive.dtype(),
        }
    }

    fn zeros(&self, shape: Shape) -> FloatTensor<B> {
        B::float_zeros(shape, &self.device, self.dtype.into())
    }
}

impl Step for CollectiveOutputStep {
    fn step(self: Box<Self>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {}

    fn depth(&self) -> usize {
        self.node.order
    }

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
