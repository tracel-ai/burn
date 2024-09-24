use alloc::{vec, vec::Vec};
use std::marker::PhantomData;

use crate::{
    checkpoint::{
        base::Checkpointer, builder::CheckpointerBuilder, retro_forward::RetroForward,
        state::BackwardStates, strategy::CheckpointStrategy,
    },
    grads::Gradients,
    graph::{ComputingProperty, NodeID, NodeRef, Requirement, Step},
    ops::{binary, broadcast_shape, unary, Backward, Ops, OpsKind},
    retro_binary, retro_unary, retro_unary_scalar,
    tensor::AutodiffTensor,
    utils::duplicate,
    Autodiff,
};

use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, IntTensor},
    Device, ElementConversion, Shape, TensorData,
};

use super::maxmin::MaxMinDim;

// Unsqueeze op on primitive.
fn unsqueeze_like<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    shape: Shape,
) -> B::FloatTensorPrimitive {
    /*
    let mut dims = [1; D2];
    let num_ones = D2 - D;
    let shape = self.shape();

    dims[num_ones..(D + num_ones)].copy_from_slice(&shape.dims[..D]);

    let shape = Shape::new(dims);
    self.reshape(shape)
    */

    let ndims_out = shape.num_dims();
    let shape = B::float_shape(&tensor);
    let ndims_in = shape.num_dims();

    let mut dims = vec![1; ndims_out];
    let num_ones = ndims_out - ndims_in;
    dims[num_ones..(ndims_in + num_ones)].copy_from_slice(&shape.dims[..ndims_in]);

    B::float_reshape(tensor, Shape::from(dims))
}

impl<B: Backend, C: CheckpointStrategy> FloatTensorOps<Self> for Autodiff<B, C> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        AutodiffTensor::new(B::float_from_data(data, device))
    }

    fn float_random(
        shape: Shape,
        distribution: burn_tensor::Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        AutodiffTensor::new(B::float_random(shape, distribution, device))
    }

    fn float_zeros(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        AutodiffTensor::new(B::float_zeros(shape, device))
    }

    fn float_ones(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        AutodiffTensor::new(B::float_ones(shape, device))
    }

    fn float_shape(tensor: &FloatTensor<Self>) -> Shape {
        B::float_shape(&tensor.primitive)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        B::float_into_data(tensor.primitive).await
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        B::float_device(&tensor.primitive)
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct ToDevice;

        impl<B: Backend> Backward<B, 1> for ToDevice {
            type State = B::Device;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_to_device(grad, &ops.state)
                });
            }
        }

        match ToDevice
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let device_old = B::float_device(&tensor.primitive);
                prep.finish(device_old, B::float_to_device(tensor.primitive, device))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_to_device(tensor.primitive, device)),
        }
    }

    fn float_empty(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        AutodiffTensor::new(B::float_empty(shape, device))
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Add;

        retro_binary!(RetroAdd, B::float_add);

        impl<B: Backend> Backward<B, 2> for Add {
            type State = (Shape, Shape);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (shape_lhs, shape_rhs) = ops.state;

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| broadcast_shape::<B>(grad, &shape_lhs),
                    |grad| broadcast_shape::<B>(grad, &shape_rhs),
                );
            }
        }

        match Add
            .prepare::<C>([lhs.node.clone(), rhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroAdd::<B>::new(lhs.node.id, rhs.node.id))
            .parents([&lhs, &rhs])
            .stateful()
        {
            OpsKind::Tracked(preps) => preps.finish(
                (
                    B::float_shape(&lhs.primitive),
                    B::float_shape(&rhs.primitive),
                ),
                B::float_add(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(preps) => preps.finish(B::float_add(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct AddScalar;

        retro_unary_scalar!(RetroAddScalar, B::float_add_scalar);

        impl<B: Backend> Backward<B, 1> for AddScalar {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
            }
        }

        AddScalar
            .prepare::<C>([lhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroAddScalar::<B>::new(lhs.node.id, rhs))
            .parents([&lhs])
            .stateless(B::float_add_scalar(lhs.primitive, rhs))
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Sub;

        retro_binary!(RetroSub, B::float_sub);

        impl<B: Backend> Backward<B, 2> for Sub {
            type State = (Shape, Shape);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (shape_lhs, shape_rhs) = ops.state;

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| broadcast_shape::<B>(grad, &shape_lhs),
                    |grad| broadcast_shape::<B>(B::float_neg(grad), &shape_rhs),
                );
            }
        }

        match Sub
            .prepare::<C>([lhs.node.clone(), rhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroSub::<B>::new(lhs.node.id, rhs.node.id))
            .parents([&lhs, &rhs])
            .stateful()
        {
            OpsKind::Tracked(preps) => preps.finish(
                (
                    B::float_shape(&lhs.primitive),
                    B::float_shape(&rhs.primitive),
                ),
                B::float_sub(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(preps) => preps.finish(B::float_sub(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SubScalar;

        retro_unary_scalar!(RetroSubScalar, B::float_sub_scalar);

        impl<B: Backend> Backward<B, 1> for SubScalar {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
            }
        }

        SubScalar
            .prepare::<C>([lhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroSubScalar::<B>::new(lhs.node.id, rhs))
            .parents([&lhs])
            .stateless(B::float_sub_scalar(lhs.primitive, rhs))
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Mul;

        retro_binary!(RetroMul, B::float_mul);

        impl<B: Backend> Backward<B, 2> for Mul {
            type State = (Option<NodeID>, Option<NodeID>, BinaryOpsBroadcast);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (lhs, rhs, broadcast) = ops.state;
                let lhs = lhs.map(|lhs| checkpointer.retrieve_node_output(lhs));
                let rhs = rhs.map(|rhs| checkpointer.retrieve_node_output(rhs));

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let grad = B::float_mul(grad, rhs.unwrap());
                        broadcast.backward_lhs::<B>(grad)
                    },
                    |grad| {
                        let grad = B::float_mul(grad, lhs.unwrap());
                        broadcast.backward_rhs::<B>(grad)
                    },
                );
            }
        }

        let lhs_tracked = lhs.is_tracked();
        let rhs_tracked = rhs.is_tracked();
        let broadcast = BinaryOpsBroadcast::new::<B>(&lhs.primitive, &rhs.primitive);

        match Mul
            .prepare::<C>([lhs.node.clone(), rhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroMul::<B>::new(lhs.node.id, rhs.node.id))
            .parents([&lhs, &rhs])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let lhs_state = rhs_tracked.then(|| prep.checkpoint(&lhs));
                let rhs_state = lhs_tracked.then(|| prep.checkpoint(&rhs));

                prep.finish(
                    (lhs_state, rhs_state, broadcast),
                    B::float_mul(lhs.primitive, rhs.primitive),
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_mul(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct MulScalar;

        retro_unary_scalar!(RetroMulScalar, B::float_mul_scalar);

        impl<B: Backend> Backward<B, 1> for MulScalar {
            type State = FloatElem<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_mul_scalar(grad, ops.state)
                });
            }
        }

        match MulScalar
            .prepare::<C>([lhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroMulScalar::<B>::new(lhs.node.id, rhs))
            .parents([&lhs])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(rhs, B::float_mul_scalar(lhs.primitive, rhs)),
            OpsKind::UnTracked(prep) => prep.finish(B::float_mul_scalar(lhs.primitive, rhs)),
        }
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Div;

        retro_binary!(RetroDiv, B::float_div);

        impl<B: Backend> Backward<B, 2> for Div {
            type State = (Option<NodeID>, Option<NodeID>, BinaryOpsBroadcast);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (lhs, rhs, broadcast) = ops.state;
                let lhs = lhs.map(|lhs| checkpointer.retrieve_node_output(lhs));
                let rhs = rhs.map(|rhs| checkpointer.retrieve_node_output(rhs));
                let [rhs_4lhs, rhs_4rhs] = duplicate(&ops.parents, rhs);

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let rhs = rhs_4lhs.unwrap();
                        let value = B::float_powf_scalar(rhs, -1.0);
                        let grad = B::float_mul(grad, value);

                        broadcast.backward_lhs::<B>(grad)
                    },
                    |grad| {
                        let rhs = rhs_4rhs.unwrap();
                        let lhs = lhs.unwrap();
                        let value = B::float_div(B::float_neg(lhs), B::float_powf_scalar(rhs, 2.0));
                        let grad = B::float_mul(grad, value);

                        broadcast.backward_rhs::<B>(grad)
                    },
                );
            }
        }

        let lhs_tracked = lhs.is_tracked();
        let rhs_tracked = rhs.is_tracked();
        let broadcast = BinaryOpsBroadcast::new::<B>(&lhs.primitive, &rhs.primitive);

        match Div
            .prepare::<C>([lhs.node.clone(), rhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroDiv::<B>::new(lhs.node.id, rhs.node.id))
            .parents([&lhs, &rhs])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let lhs_state = rhs_tracked.then(|| prep.checkpoint(&lhs));
                let rhs_state = (lhs_tracked || rhs_tracked).then(|| prep.checkpoint(&rhs));

                prep.finish(
                    (lhs_state, rhs_state, broadcast),
                    B::float_div(lhs.primitive, rhs.primitive),
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_div(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct DivScalar;

        retro_unary_scalar!(RetroDivScalar, B::float_div_scalar);

        impl<B: Backend> Backward<B, 1> for DivScalar {
            type State = FloatElem<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let tmp = 1.0 / ops.state.elem::<f32>();
                    B::float_mul_scalar(grad, tmp.elem())
                });
            }
        }

        match DivScalar
            .prepare::<C>([lhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroDivScalar::<B>::new(lhs.node.id, rhs))
            .parents([&lhs])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(rhs, B::float_div_scalar(lhs.primitive, rhs)),
            OpsKind::UnTracked(prep) => prep.finish(B::float_div_scalar(lhs.primitive, rhs)),
        }
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct RemainderScalar;

        retro_unary_scalar!(RetroRemainderScalar, B::float_remainder_scalar);

        impl<B: Backend> Backward<B, 1> for RemainderScalar {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
            }
        }

        RemainderScalar
            .prepare::<C>([lhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroRemainderScalar::<B>::new(lhs.node.id, rhs))
            .parents([&lhs])
            .stateless(B::float_remainder_scalar(lhs.primitive, rhs))
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Matmul;

        impl<B: Backend> Backward<B, 2> for Matmul {
            type State = (Option<NodeID>, Option<NodeID>, BinaryOpsBroadcast);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (lhs, rhs, broadcast) = ops.state;
                let lhs = lhs.map(|lhs| checkpointer.retrieve_node_output(lhs));
                let rhs = rhs.map(|rhs| checkpointer.retrieve_node_output(rhs));

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let rhs = B::float_transpose(rhs.unwrap());
                        let grad = B::float_matmul(grad, rhs);

                        broadcast.backward_lhs::<B>(grad)
                    },
                    |grad| {
                        let lhs = B::float_transpose(lhs.unwrap());
                        let grad = B::float_matmul(lhs, grad);

                        broadcast.backward_rhs::<B>(grad)
                    },
                );
            }
        }

        let lhs_tracked = lhs.is_tracked();
        let rhs_tracked = rhs.is_tracked();
        let broadcast = BinaryOpsBroadcast::new::<B>(&lhs.primitive, &rhs.primitive);

        match Matmul
            .prepare::<C>([lhs.node.clone(), rhs.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let lhs_state = rhs_tracked.then(|| prep.checkpoint(&lhs));
                let rhs_state = lhs_tracked.then(|| prep.checkpoint(&rhs));
                prep.finish(
                    (lhs_state, rhs_state, broadcast),
                    B::float_matmul(lhs.primitive, rhs.primitive),
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_matmul(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Neg;

        retro_unary!(RetroNeg, B::float_neg);

        impl<B: Backend> Backward<B, 1> for Neg {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| B::float_neg(grad));
            }
        }

        Neg.prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroNeg::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateless(B::float_neg(tensor.primitive))
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Recip;

        retro_unary!(RetroRecip, B::float_recip);

        impl<B: Backend> Backward<B, 1> for Recip {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let tensor = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let tmp = B::float_powf_scalar(tensor, -2.0);
                    let value = B::float_neg(tmp);

                    B::float_mul(grad, value)
                });
            }
        }

        match Recip
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroRecip::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_recip(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_recip(tensor.primitive)),
        }
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SwapDim;

        #[derive(new, Debug)]
        struct RetroSwapDims<B: Backend> {
            input_id: NodeID,
            dim1: usize,
            dim2: usize,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroSwapDims<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
                let out = B::float_swap_dims(input, self.dim1, self.dim2);
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for SwapDim {
            type State = (usize, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (dim1, dim2) = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_swap_dims(grad, dim2, dim1)
                });
            }
        }

        match SwapDim
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSwapDims::<B>::new(tensor.node.id, dim1, dim2))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (dim1, dim2),
                B::float_swap_dims(tensor.primitive, dim1, dim2),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::float_swap_dims(tensor.primitive, dim1, dim2))
            }
        }
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct PermuteDim;

        #[derive(new, Debug)]
        struct RetroPermuteDims<B: Backend> {
            input_id: NodeID,
            axes: Vec<usize>,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroPermuteDims<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
                let out = B::float_permute(input, &self.axes);
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for PermuteDim {
            type State = Vec<usize>;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let axes = ops.state;

                let mut inverse = vec![0usize; axes.len()];
                axes.iter()
                    .enumerate()
                    .for_each(|(i, &axis)| inverse[axis] = i);

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_permute(grad, &inverse)
                });
            }
        }

        match PermuteDim
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroPermuteDims::<B>::new(tensor.node.id, axes.to_vec()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                prep.finish(axes.to_vec(), B::float_permute(tensor.primitive, axes))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_permute(tensor.primitive, axes)),
        }
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct FlipDim;

        #[derive(new, Debug)]
        struct RetroFlipDims<B: Backend> {
            input_id: NodeID,
            axes: Vec<usize>,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroFlipDims<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
                let out = B::float_flip(input, &self.axes);
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for FlipDim {
            type State = Vec<usize>;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let axes = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_flip(grad, &axes)
                });
            }
        }

        match FlipDim
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroFlipDims::<B>::new(tensor.node.id, axes.to_vec()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                prep.finish(axes.to_vec(), B::float_flip(tensor.primitive, axes))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_flip(tensor.primitive, axes)),
        }
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct ReshapeDim;

        #[derive(new, Debug)]
        struct RetroReshape<B: Backend> {
            input_id: NodeID,
            shape: Shape,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroReshape<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
                let out = B::float_reshape(input, self.shape.clone());
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for ReshapeDim {
            type State = (Shape, Shape);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (shape_original, shape) = ops.state;
                let ndims_out = shape.num_dims();

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let shape_grad = B::float_shape(&grad);
                    let mut grad = grad;

                    for i in 0..ndims_out {
                        if shape.dims[i] == 1 && shape_grad.dims[i] != 1 {
                            grad = B::float_sum_dim(grad, i);
                        }
                    }

                    B::float_reshape(grad, shape_original)
                });
            }
        }

        match ReshapeDim
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroReshape::<B>::new(tensor.node.id, shape.clone()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (B::float_shape(&tensor.primitive), shape.clone()),
                B::float_reshape(tensor.primitive, shape),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_reshape(tensor.primitive, shape)),
        }
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<B>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Gather;

        impl<B: Backend> Backward<B, 1> for Gather {
            type State = (usize, IntTensor<B>, Shape, B::Device);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (dim, indices, shape, device) = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let zeros = B::float_zeros(shape, &device);
                    B::float_scatter(dim, zeros, indices, grad)
                });
            }
        }

        match Gather
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    dim,
                    indices.clone(),
                    B::float_shape(&tensor.primitive),
                    B::float_device(&tensor.primitive),
                ),
                B::float_gather(dim, tensor.primitive, indices),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::float_gather(dim, tensor.primitive, indices))
            }
        }
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<B>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Scatter;

        impl<B: Backend> Backward<B, 2> for Scatter {
            type State = (usize, IntTensor<B>, Shape, Shape, B::Device);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (dim, indices, shape_lhs, shape_rhs, device) = ops.state;
                let [indices_4lhs, indices_4rhs] = duplicate(&ops.parents, Some(indices));

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let zeros = B::float_zeros(shape_lhs, &device);
                        B::float_scatter(dim, grad, indices_4lhs.unwrap(), zeros)
                    },
                    |grad| {
                        let zeros = B::float_zeros(shape_rhs, &device);
                        B::float_scatter(dim, zeros, indices_4rhs.unwrap(), grad)
                    },
                );
            }
        }

        match Scatter
            .prepare::<C>([tensor.node, value.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    dim,
                    indices.clone(),
                    B::float_shape(&tensor.primitive),
                    B::float_shape(&value.primitive),
                    B::float_device(&value.primitive),
                ),
                B::float_scatter(dim, tensor.primitive, indices, value.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_scatter(
                dim,
                tensor.primitive,
                indices,
                value.primitive,
            )),
        }
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<B>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Select;

        #[derive(new, Debug)]
        struct RetroSelect<B: Backend> {
            input_id: NodeID,
            dim: usize,
            indices: IntTensor<B>,
        }

        impl<B: Backend> RetroForward for RetroSelect<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
                let out = B::float_select(input, self.dim, self.indices.clone());
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for Select {
            type State = (usize, IntTensor<B>, Shape, B::Device);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (dim, indices, shape, device) = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let zeros = B::float_zeros(shape, &device);
                    B::float_select_assign(zeros, dim, indices, grad)
                });
            }
        }

        match Select
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSelect::<B>::new(tensor.node.id, dim, indices.clone()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    dim,
                    indices.clone(),
                    B::float_shape(&tensor.primitive),
                    B::float_device(&tensor.primitive),
                ),
                B::float_select(tensor.primitive, dim, indices),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::float_select(tensor.primitive, dim, indices))
            }
        }
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<B>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct IndexSelectDimAssign;

        #[derive(new, Debug)]
        struct RetroSelectAssign<B: Backend> {
            tensor_id: NodeID,
            dim: usize,
            indices: IntTensor<B>,
            value_id: NodeID,
        }

        impl<B: Backend> RetroForward for RetroSelectAssign<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor = states.get_state::<B::FloatTensorPrimitive>(&self.tensor_id);
                let value = states.get_state::<B::FloatTensorPrimitive>(&self.value_id);
                let out = B::float_select_assign(tensor, self.dim, self.indices.clone(), value);
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 2> for IndexSelectDimAssign {
            type State = (usize, IntTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (dim, indices) = ops.state;

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| grad,
                    |grad| B::float_select(grad, dim, indices),
                );
            }
        }

        match IndexSelectDimAssign
            .prepare::<C>([tensor.node.clone(), value.node.clone()])
            .memory_bound()
            .retro_forward(RetroSelectAssign::<B>::new(
                tensor.node.id,
                dim,
                indices.clone(),
                value.node.id,
            ))
            .parents([&tensor, &value])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (dim, indices.clone()),
                B::float_select_assign(tensor.primitive, dim, indices, value.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_select_assign(
                tensor.primitive,
                dim,
                indices,
                value.primitive,
            )),
        }
    }

    fn float_slice(
        tensor: FloatTensor<Self>,
        ranges: &[std::ops::Range<usize>],
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Index;

        #[derive(new, Debug)]
        struct RetroSlice<B: Backend> {
            tensor_id: NodeID,
            ranges: Vec<std::ops::Range<usize>>,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroSlice<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor = states.get_state::<B::FloatTensorPrimitive>(&self.tensor_id);
                let out = B::float_slice(tensor, &self.ranges);
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for Index {
            type State = (Vec<std::ops::Range<usize>>, Shape, B::Device);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (ranges, shape, device) = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let zeros = B::float_zeros(shape, &device);
                    B::float_slice_assign(zeros, &ranges, grad)
                });
            }
        }

        match Index
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSlice::<B>::new(tensor.node.id, ranges.to_vec()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    ranges.to_vec(),
                    B::float_shape(&tensor.primitive),
                    B::float_device(&tensor.primitive),
                ),
                B::float_slice(tensor.primitive, ranges),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_slice(tensor.primitive, ranges)),
        }
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[std::ops::Range<usize>],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SliceAssign;

        #[derive(new, Debug)]
        struct RetroSliceAssign<B: Backend> {
            tensor_id: NodeID,
            ranges: Vec<std::ops::Range<usize>>,
            value_id: NodeID,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroSliceAssign<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor = states.get_state::<B::FloatTensorPrimitive>(&self.tensor_id);
                let value = states.get_state::<B::FloatTensorPrimitive>(&self.value_id);
                let out = B::float_slice_assign(tensor, &self.ranges, value);
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 2> for SliceAssign {
            type State = (Vec<std::ops::Range<usize>>, Shape, B::Device);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (ranges, shape_rhs, device) = ops.state;
                let [ranges_4lhs, ranges_4rhs] = duplicate(&ops.parents, Some(ranges));

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let zeros = B::float_zeros(shape_rhs, &device);
                        B::float_slice_assign(grad, &ranges_4lhs.unwrap(), zeros)
                    },
                    |grad| B::float_slice(grad, &ranges_4rhs.unwrap()),
                );
            }
        }

        match SliceAssign
            .prepare::<C>([tensor.node.clone(), value.node.clone()])
            .memory_bound()
            .retro_forward(RetroSliceAssign::<B>::new(
                tensor.node.id,
                ranges.to_vec(),
                value.node.id,
            ))
            .parents([&tensor, &value])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    ranges.to_vec(),
                    B::float_shape(&value.primitive),
                    B::float_device(&value.primitive),
                ),
                B::float_slice_assign(tensor.primitive, ranges, value.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_slice_assign(
                tensor.primitive,
                ranges,
                value.primitive,
            )),
        }
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        source: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct MaskWhere;

        impl<B: Backend> Backward<B, 2> for MaskWhere {
            type State = (BoolTensor<B>, Shape, Shape, B::Device);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (mask, shape_lhs, shape_rhs, device) = ops.state;
                let [mask_4lhs, mask_4rhs] = duplicate(&ops.parents, Some(mask));

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let zeros = B::float_zeros(shape_lhs.clone(), &device);
                        let grad = B::float_mask_where(grad, mask_4lhs.unwrap(), zeros);

                        broadcast_shape::<B>(grad, &shape_lhs)
                    },
                    |grad| {
                        let zeros = B::float_zeros(shape_rhs.clone(), &device);
                        let grad = B::float_mask_where(zeros, mask_4rhs.unwrap(), grad);

                        broadcast_shape::<B>(grad, &shape_rhs)
                    },
                );
            }
        }

        match MaskWhere
            .prepare::<C>([tensor.node, source.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    mask.clone(),
                    B::float_shape(&tensor.primitive),
                    B::float_shape(&source.primitive),
                    B::float_device(&source.primitive),
                ),
                B::float_mask_where(tensor.primitive, mask, source.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_mask_where(
                tensor.primitive,
                mask,
                source.primitive,
            )),
        }
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<B>,
        value: FloatElem<B>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct MaskFill;

        impl<B: Backend> Backward<B, 1> for MaskFill {
            type State = BoolTensor<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_mask_fill(grad, ops.state, 0.elem())
                });
            }
        }

        match MaskFill
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                mask.clone(),
                B::float_mask_fill(tensor.primitive, mask, value),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::float_mask_fill(tensor.primitive, mask, value))
            }
        }
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<B> {
        B::float_equal(lhs.primitive, rhs.primitive)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> BoolTensor<B> {
        B::float_equal_elem(lhs.primitive, rhs)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<B> {
        B::float_greater(lhs.primitive, rhs.primitive)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> BoolTensor<B> {
        B::float_greater_elem(lhs.primitive, rhs)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<B> {
        B::float_greater_equal(lhs.primitive, rhs.primitive)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> BoolTensor<B> {
        B::float_greater_equal_elem(lhs.primitive, rhs)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<B> {
        B::float_lower(lhs.primitive, rhs.primitive)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> BoolTensor<B> {
        B::float_lower_elem(lhs.primitive, rhs)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<B> {
        B::float_lower_equal(lhs.primitive, rhs.primitive)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<B>) -> BoolTensor<B> {
        B::float_lower_equal_elem(lhs.primitive, rhs)
    }

    fn float_detach(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // When we detach a tensor, we remove it from the graph, but we still want to keep the
        // `require_grad` setting.
        let is_require_grad = Self::float_is_require_grad(&tensor);
        let tensor = AutodiffTensor::new(tensor.primitive);

        match is_require_grad {
            true => tensor.require_grad(),
            false => tensor,
        }
    }

    fn float_set_require_grad(tensor: FloatTensor<Self>, require_grad: bool) -> FloatTensor<Self> {
        if require_grad {
            return tensor.require_grad();
        }

        AutodiffTensor::new(tensor.primitive)
    }

    fn float_is_require_grad(tensor: &FloatTensor<Self>) -> bool {
        matches!(tensor.node.requirement, Requirement::Grad)
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Mean;

        impl<B: Backend> Backward<B, 1> for Mean {
            type State = Shape;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let shape = ops.state;
                    let val = 1_f64 / shape.num_elements() as f64;
                    let ones = B::float_ones(shape, &B::float_device(&grad));
                    let val = B::float_mul_scalar(ones, val.elem());

                    let grad = unsqueeze_like::<B>(grad, B::float_shape(&val));
                    B::float_mul(val, grad)
                });
            }
        }

        match Mean.prepare::<C>([tensor.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(
                B::float_shape(&tensor.primitive),
                B::float_mean(tensor.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_mean(tensor.primitive)),
        }
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Sum;

        impl<B: Backend> Backward<B, 1> for Sum {
            type State = Shape;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let val = B::float_ones(ops.state, &B::float_device(&grad));

                    let grad = unsqueeze_like::<B>(grad, B::float_shape(&val));
                    B::float_mul(val, grad)
                });
            }
        }

        match Sum.prepare::<C>([tensor.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(
                B::float_shape(&tensor.primitive),
                B::float_sum(tensor.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_sum(tensor.primitive)),
        }
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct MeanDim;

        impl<B: Backend> Backward<B, 1> for MeanDim {
            type State = (Shape, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (shape, dim) = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let val = 1_f64 / shape.dims[dim] as f64;
                    let ones = B::float_ones(shape, &B::float_device(&grad));
                    let val = B::float_mul_scalar(ones, B::FloatElem::from_elem(val));

                    let grad = B::float_sum_dim(grad, dim);
                    B::float_mul(val, grad)
                });
            }
        }

        match MeanDim
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (B::float_shape(&tensor.primitive), dim),
                B::float_mean_dim(tensor.primitive, dim),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_mean_dim(tensor.primitive, dim)),
        }
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SumDim;

        impl<B: Backend> Backward<B, 1> for SumDim {
            type State = (Shape, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (shape, dim) = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let ones = B::float_ones(shape, &B::float_device(&grad));
                    let grad = B::float_sum_dim(grad, dim);

                    B::float_mul(ones, grad)
                });
            }
        }

        match SumDim
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (B::float_shape(&tensor.primitive), dim),
                B::float_sum_dim(tensor.primitive, dim),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_sum_dim(tensor.primitive, dim)),
        }
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<B> {
        B::float_argmax(tensor.primitive, dim)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<B> {
        B::float_argmin(tensor.primitive, dim)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Exp;

        retro_unary!(RetroExp, B::float_exp);

        impl<B: Backend> Backward<B, 1> for Exp {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                let output = B::float_exp(input);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_mul(grad, output)
                });
            }
        }

        match Exp
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroExp::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_exp(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_exp(tensor.primitive)),
        }
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Log;

        retro_unary!(RetroLog, B::float_log);

        impl<B: Backend> Backward<B, 1> for Log {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::float_powf_scalar(input, -1.0);
                    B::float_mul(grad, value)
                });
            }
        }

        match Log
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroLog::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_log(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_log(tensor.primitive)),
        }
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Log1P;

        retro_unary!(RetroLog1P, B::float_log1p);

        impl<B: Backend> Backward<B, 1> for Log1P {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::float_add_scalar(input, 1.elem());
                    let value = B::float_powf_scalar(value, -1.0);

                    B::float_mul(grad, value)
                });
            }
        }

        match Log1P
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroLog1P::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_log1p(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_log1p(tensor.primitive)),
        }
    }

    fn float_powf_scalar(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct PowfScalar;

        #[derive(new, Debug)]
        struct RetroPowfScalar<B: Backend> {
            lhs_id: NodeID,
            rhs: f32,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroPowfScalar<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let lhs = states.get_state::<B::FloatTensorPrimitive>(&self.lhs_id);
                let out = B::float_powf_scalar(lhs, self.rhs);
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for PowfScalar {
            type State = (NodeID, f32);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (tensor_id, value) = ops.state;
                let tensor = checkpointer.retrieve_node_output(tensor_id);

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let tmp = B::float_powf_scalar(tensor, value - 1.0);
                    let value = B::float_mul_scalar(tmp, value.elem());

                    B::float_mul(grad, value)
                });
            }
        }

        match PowfScalar
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroPowfScalar::<B>::new(tensor.node.id, value))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = (prep.checkpoint(&tensor), value);
                prep.finish(state, B::float_powf_scalar(tensor.primitive, value))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_powf_scalar(tensor.primitive, value)),
        }
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Sqrt;

        retro_unary!(RetroSqrt, B::float_sqrt);

        impl<B: Backend> Backward<B, 1> for Sqrt {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::float_div_scalar(B::float_powf_scalar(input, -0.5), 2.elem());

                    B::float_mul(grad, value)
                });
            }
        }

        match Sqrt
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSqrt::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_sqrt(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_sqrt(tensor.primitive)),
        }
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Abs;

        retro_unary!(RetroAbs, B::float_abs);

        impl<B: Backend> Backward<B, 1> for Abs {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let tensor: B::FloatTensorPrimitive = checkpointer.retrieve_node_output(ops.state);
                let state = B::float_sign(tensor);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_mul(grad, state)
                });
            }
        }

        match Abs
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroAbs::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_abs(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_abs(tensor.primitive)),
        }
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Cos;

        retro_unary!(RetroCos, B::float_cos);

        impl<B: Backend> Backward<B, 1> for Cos {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::float_neg(B::float_sin(input));

                    B::float_mul(grad, value)
                });
            }
        }

        match Cos
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroCos::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_cos(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_cos(tensor.primitive)),
        }
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Sin;

        retro_unary!(RetroSin, B::float_sin);

        impl<B: Backend> Backward<B, 1> for Sin {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let state = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::float_cos(state);
                    B::float_mul(grad, value)
                });
            }
        }

        match Sin
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSin::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_sin(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_sin(tensor.primitive)),
        }
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Tanh;

        retro_unary!(RetroTanh, B::float_tanh);

        impl<B: Backend> Backward<B, 1> for Tanh {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                let state = B::float_tanh(input);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::float_add_scalar(
                        B::float_neg(B::float_powf_scalar(state, 2.0)),
                        1.elem(),
                    );
                    B::float_mul(grad, value)
                });
            }
        }

        match Tanh
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroTanh::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_tanh(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_tanh(tensor.primitive)),
        }
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Erf;

        retro_unary!(RetroErf, B::float_erf);

        impl<B: Backend> Backward<B, 1> for Erf {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let ops = checkpointer.retrieve_node_output(ops.state);
                    let exponent = B::float_neg(B::float_powf_scalar(ops, 2.0));
                    let numerator = B::float_mul_scalar(B::float_exp(exponent), 2.0.elem());
                    let denominator = std::f64::consts::PI.sqrt().elem();
                    let value = B::float_div_scalar(numerator, denominator);

                    B::float_mul(grad, value)
                });
            }
        }

        match Erf
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroErf::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::float_erf(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_erf(tensor.primitive)),
        }
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CatStep<B: Backend> {
            nodes: Vec<Option<NodeRef>>,
            // The dimension of each tensor along the dim dimension.
            // This indicates the number of dimension concatenated for each tensor.
            dim_sizes: Vec<usize>,
            output: NodeRef,
            phantom: PhantomData<B>,
            dim: usize,
        }

        impl<B: Backend> Step for CatStep<B> {
            fn step(self: Box<Self>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
                let grad = grads.consume::<B>(&self.output);
                let ranges: Vec<_> = B::float_shape(&grad).dims.iter().map(|v| 0..*v).collect();

                let mut current_index = 0;

                self.nodes
                    .into_iter()
                    .zip(self.dim_sizes)
                    .filter_map(|(node, dim_size)| node.map(|node| (node, dim_size)))
                    .for_each(|(node, dim_size)| {
                        let mut ranges = ranges.clone();
                        ranges[self.dim] = current_index..dim_size + current_index;
                        current_index += dim_size;
                        grads.register::<B>(node.id, B::float_slice(grad.clone(), &ranges));
                    });
            }

            fn node(&self) -> NodeID {
                self.output.id
            }

            fn parents(&self) -> Vec<NodeID> {
                self.nodes
                    .iter()
                    .filter_map(|node| node.clone())
                    .map(|node| node.id)
                    .collect()
            }
            fn depth(&self) -> usize {
                self.output.order
            }
        }

        let mut nodes = Vec::with_capacity(tensors.len());
        let mut primitives = Vec::with_capacity(tensors.len());
        let mut dim_sizes = Vec::with_capacity(tensors.len());

        tensors.into_iter().for_each(|tensor| {
            dim_sizes.push(B::float_shape(&tensor.primitive).dims[dim]);
            nodes.push(tensor.node);
            primitives.push(tensor.primitive);
        });

        let requirement = Requirement::from_nodes(&nodes);

        // For simplicity, this operation does not checkpoint anything
        let cat_computing_property = ComputingProperty::Ambiguous;
        let checkpointer_builder = CheckpointerBuilder::default();

        let output = B::float_cat(primitives, dim);
        if requirement.is_none() {
            return AutodiffTensor::from_parents(
                output,
                &nodes,
                requirement,
                cat_computing_property,
            );
        }

        let output =
            AutodiffTensor::from_parents(output, &nodes, requirement, cat_computing_property);
        let nodes = nodes
            .into_iter()
            .map(|node| node.clone_if_require_grad())
            .collect::<Vec<_>>();

        let ops = CatStep::<B>::new(nodes, dim_sizes, output.node.clone(), dim);
        output.register_step(ops, checkpointer_builder)
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        match MaxMinDim
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let shape = B::float_shape(&tensor.primitive);
                let (tensor, index) = B::float_max_dim_with_indices(tensor.primitive, dim);
                prep.finish((index, shape), tensor)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_max_dim(tensor.primitive, dim)),
        }
    }
    fn float_max_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<B>) {
        match MaxMinDim
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let shape = B::float_shape(&tensor.primitive);
                let (tensor, index) = B::float_max_dim_with_indices(tensor.primitive, dim);
                let tensor = prep.finish((index.clone(), shape), tensor);

                (tensor, index)
            }
            OpsKind::UnTracked(prep) => {
                let (tensor, index) = B::float_max_dim_with_indices(tensor.primitive, dim);
                let tensor = prep.finish(tensor);

                (tensor, index)
            }
        }
    }
    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        match MaxMinDim
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let shape = B::float_shape(&tensor.primitive);
                let (tensor, index) = B::float_min_dim_with_indices(tensor.primitive, dim);
                prep.finish((index, shape), tensor)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_min_dim(tensor.primitive, dim)),
        }
    }
    fn float_min_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<B>) {
        match MaxMinDim
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let shape = B::float_shape(&tensor.primitive);
                let (tensor, index) = B::float_min_dim_with_indices(tensor.primitive, dim);
                let tensor = prep.finish((index.clone(), shape), tensor);

                (tensor, index)
            }
            OpsKind::UnTracked(prep) => {
                let (tensor, index) = B::float_min_dim_with_indices(tensor.primitive, dim);
                let tensor = prep.finish(tensor);

                (tensor, index)
            }
        }
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> <Autodiff<B> as Backend>::IntTensorPrimitive {
        B::float_into_int(tensor.primitive)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct PowF;

        retro_binary!(RetroPowf, B::float_powf);

        impl<B: Backend> Backward<B, 2> for PowF {
            type State = (NodeID, NodeID, BinaryOpsBroadcast);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (lhs_id, rhs_id, broadcast) = ops.state;
                let lhs: B::FloatTensorPrimitive = checkpointer.retrieve_node_output(lhs_id);
                let rhs: B::FloatTensorPrimitive = checkpointer.retrieve_node_output(rhs_id);

                // Both lhs and rhs are needed for both lhs and rhs gradients, but we clone them
                // the number of times required by the parents specification.
                let [rhs_4lhs, rhs_4rhs] = duplicate(&ops.parents, Some(rhs));
                let [lhs_4lhs, lhs_4rhs] = duplicate(&ops.parents, Some(lhs));

                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        //rhs*(lhs.val**(rhs-1))*grad
                        let rhs1 = rhs_4lhs.unwrap();
                        let rhs2 = rhs1.clone();
                        let lhs = lhs_4lhs.unwrap();

                        let tmp = B::float_powf(
                            lhs,
                            B::float_sub_scalar(rhs1, B::FloatElem::from_elem(1.0)),
                        );
                        let value = B::float_mul(tmp, rhs2);
                        let grad = B::float_mul(grad, value);

                        broadcast.backward_lhs::<B>(grad)
                    },
                    |grad| {
                        //lhs**rhs * ln(lhs) * grad
                        let rhs = rhs_4rhs.unwrap();
                        let lhs1 = lhs_4rhs.unwrap();
                        let lhs2 = lhs1.clone();
                        let tmp = B::float_powf(lhs1, rhs);
                        let value = B::float_mul(tmp, B::float_log(lhs2));
                        let grad = B::float_mul(grad, value);

                        broadcast.backward_rhs::<B>(grad)
                    },
                );
            }
        }

        let broadcast = BinaryOpsBroadcast::new::<B>(&lhs.primitive, &rhs.primitive);

        match PowF
            .prepare::<C>([lhs.node.clone(), rhs.node.clone()])
            .memory_bound()
            .retro_forward(RetroPowf::<B>::new(lhs.node.id, rhs.node.id))
            .parents([&lhs, &rhs])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let lhs_state = prep.checkpoint(&lhs);
                let rhs_state = prep.checkpoint(&rhs);
                prep.finish(
                    (lhs_state, rhs_state, broadcast),
                    B::float_powf(lhs.primitive, rhs.primitive),
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::float_powf(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Sign;

        retro_unary!(RetroSign, B::float_sign);

        impl<B: Backend> Backward<B, 1> for Sign {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad|
                        // Always return 0 because the derivative of the sign function
                        // does not contribute to gradient updates in a meaningful way.
                        B::float_mul_scalar(grad, 0.elem()));
            }
        }

        Sign.prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSign::<B>::new(tensor.node.id))
            .parents([&tensor])
            .stateless(B::float_sign(tensor.primitive))
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        // D1: tensor, D2: shape
        #[derive(Debug)]
        struct ExpandDim;

        #[derive(new, Debug)]
        struct RetroExpand<B: Backend> {
            input_id: NodeID,
            shape: Shape,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroExpand<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
                let out = B::float_expand(input, self.shape.clone());
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for ExpandDim {
            type State = (Shape, Shape);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (shape_in, shape_out) = ops.state;
                let ndims_in = shape_in.num_dims();
                let ndims_out = shape_out.num_dims();

                let mut shape_expanded = vec![1; ndims_out];

                debug_assert!(ndims_out >= ndims_in);

                for i in 0..ndims_in {
                    shape_expanded[i + (ndims_out - ndims_in)] = shape_in.dims[i];
                }

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let shape_grad = B::float_shape(&grad);
                    let mut grad = grad;

                    #[allow(clippy::needless_range_loop)]
                    for i in 0..ndims_out {
                        if shape_expanded[i] == 1 && shape_grad.dims[i] != 1 {
                            grad = B::float_sum_dim(grad, i);
                        }
                    }

                    B::float_reshape(grad, shape_in)
                });
            }
        }

        match ExpandDim
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroExpand::<B>::new(tensor.node.id, shape.clone()))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (B::float_shape(&tensor.primitive), shape.clone()),
                B::float_expand(tensor.primitive, shape),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_expand(tensor.primitive, shape)),
        }
    }

    fn float_sort(tensor: FloatTensor<Self>, dim: usize, descending: bool) -> FloatTensor<Self> {
        match super::sort::SortDim
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let shape = B::float_shape(&tensor.primitive);
                let (tensor, indices) =
                    B::float_sort_with_indices(tensor.primitive, dim, descending);
                prep.finish((indices, shape), tensor)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::float_sort(tensor.primitive, dim, descending))
            }
        }
    }

    fn float_sort_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> (FloatTensor<Self>, IntTensor<B>) {
        match super::sort::SortDim
            .prepare::<C>([tensor.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let shape = B::float_shape(&tensor.primitive);
                let (tensor, indices) =
                    B::float_sort_with_indices(tensor.primitive, dim, descending);
                let tensor = prep.finish((indices.clone(), shape), tensor);

                (tensor, indices)
            }
            OpsKind::UnTracked(prep) => {
                let (tensor, indices) =
                    B::float_sort_with_indices(tensor.primitive, dim, descending);
                let tensor = prep.finish(tensor);

                (tensor, indices)
            }
        }
    }

    fn float_argsort(tensor: FloatTensor<Self>, dim: usize, descending: bool) -> IntTensor<B> {
        B::float_argsort(tensor.primitive, dim, descending)
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Repeat;

        #[derive(new, Debug)]
        struct RetroRepeat<B: Backend> {
            tensor_id: NodeID,
            dim: usize,
            times: usize,
            _backend: PhantomData<B>,
        }

        impl<B: Backend> RetroForward for RetroRepeat<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor = states.get_state::<B::FloatTensorPrimitive>(&self.tensor_id);
                let out = B::float_repeat_dim(tensor, self.dim, self.times);
                states.save(out_node, out)
            }
        }

        impl<B: Backend> Backward<B, 1> for Repeat {
            type State = usize;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let dim = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::float_sum_dim(grad, dim)
                });
            }
        }

        match Repeat
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroRepeat::<B>::new(tensor.node.id, dim, times))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                prep.finish(dim, B::float_repeat_dim(tensor.primitive, dim, times))
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::float_repeat_dim(tensor.primitive, dim, times))
            }
        }
    }

    // TODO: Implement float_prod and float_sum
    // https://github.com/tracel-ai/burn/issues/1458
}

#[derive(Debug, Clone)]
enum BinaryOpsBroadcast {
    Broadcasted(Shape, Shape),
    None,
}

impl BinaryOpsBroadcast {
    fn new<B: Backend>(lhs: &B::FloatTensorPrimitive, rhs: &B::FloatTensorPrimitive) -> Self {
        let shape_lhs = B::float_shape(lhs);
        let shape_rhs = B::float_shape(rhs);
        let ndims = shape_lhs.num_dims();

        for i in 0..ndims {
            if shape_rhs.dims[i] != shape_lhs.dims[i] {
                return Self::Broadcasted(shape_lhs, shape_rhs);
            }
        }

        Self::None
    }

    fn backward_lhs<B: Backend>(&self, grad: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        match self {
            BinaryOpsBroadcast::Broadcasted(lhs, _rhs) => broadcast_shape::<B>(grad, lhs),
            BinaryOpsBroadcast::None => grad,
        }
    }

    fn backward_rhs<B: Backend>(&self, grad: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        match self {
            BinaryOpsBroadcast::Broadcasted(_lhs, rhs) => broadcast_shape::<B>(grad, rhs),
            BinaryOpsBroadcast::None => grad,
        }
    }
}
