use std::{marker::PhantomData, process::Output};

use crate::{
    checkpoint::{
        base::{Checkpointer, RetroForward},
        state::BackwardStates,
    },
    grads::Gradients,
    graph::{Graph, NodeID, NodeRef, Requirement, Step},
    ops::{
        binary, broadcast_shape, tensor, unary, unary_different_backend, Backward, Ops, OpsKind,
    },
    tensor::AutodiffTensor,
    utils::duplicate,
    Autodiff,
};

use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, FullPrecisionBackend, IntTensor},
    Data, Device, ElementConversion, Reader, Shape, Tensor,
};

// use super::maxmin::MaxMinDim;

impl<B: Backend> FloatTensorOps<Self> for Autodiff<B> {
    fn float_from_data<const D: usize>(
        data: Data<FloatElem<B>, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        AutodiffTensor::new(B::float_from_data(data, device))
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: burn_tensor::Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        AutodiffTensor::new(B::float_random(shape, distribution, device))
    }

    fn float_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        Self::float_from_data(Data::zeros(shape), device)
    }

    fn float_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        Self::float_from_data(Data::ones(shape), device)
    }

    fn float_shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        B::float_shape(&tensor.primitive)
    }

    fn float_to_data<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> Reader<Data<FloatElem<B>, D>> {
        B::float_to_data(&tensor.primitive)
    }

    fn float_into_data<const D: usize>(
        tensor: FloatTensor<Self, D>,
    ) -> Reader<Data<FloatElem<B>, D>> {
        B::float_into_data(tensor.primitive)
    }

    fn float_device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        B::float_device(&tensor.primitive)
    }

    fn float_to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct ToDevice;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for ToDevice {
        //     type State = B::Device;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             B::to_device(grad, &ops.state)
        //         });
        //     }
        // }

        // match ToDevice.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let device_old = B::device(&tensor.primitive);
        //         prep.finish(device_old, B::to_device(tensor.primitive, device))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::to_device(tensor.primitive, device)),
        // }
    }

    fn float_arange(range: std::ops::Range<usize>, device: &Device<Self>) -> IntTensor<Self, 1> {
        B::float_arange(range, device)
    }

    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        AutodiffTensor::new(B::float_empty(shape, device))
    }

    fn float_add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Add;

        // impl<B: Backend, const D: usize> Backward<B, D, 2> for Add {
        //     type State = (Shape<D>, Shape<D>);

        //     fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
        //         let (shape_lhs, shape_rhs) = ops.state;

        //         binary::<B, D, D, D, _, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| broadcast_shape::<B, D>(grad, &shape_lhs),
        //             |grad| broadcast_shape::<B, D>(grad, &shape_rhs),
        //         );
        //     }
        // }

        // match Add
        //     .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
        //     .stateful()
        // {
        //     OpsKind::Tracked(preps) => preps.finish(
        //         (B::shape(&lhs.primitive), B::shape(&rhs.primitive)),
        //         B::add(lhs.primitive, rhs.primitive),
        //     ),
        //     OpsKind::UnTracked(preps) => preps.finish(B::add(lhs.primitive, rhs.primitive)),
        // }
    }

    fn float_add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct AddScalar;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for AddScalar {
        //     type State = ();

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| grad);
        //     }
        // }

        // AddScalar
        //     .prepare([lhs.node], [lhs.graph])
        //     .stateless(B::add_scalar(lhs.primitive, rhs))
    }

    fn float_sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Sub;

        // impl<B: Backend, const D: usize> Backward<B, D, 2> for Sub {
        //     type State = (Shape<D>, Shape<D>);

        //     fn float_backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
        //         let (shape_lhs, shape_rhs) = ops.state;

        //         binary::<B, D, D, D, _, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| broadcast_shape::<B, D>(grad, &shape_lhs),
        //             |grad| broadcast_shape::<B, D>(B::neg(grad), &shape_rhs),
        //         );
        //     }
        // }

        // match Sub
        //     .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
        //     .stateful()
        // {
        //     OpsKind::Tracked(preps) => preps.finish(
        //         (B::shape(&lhs.primitive), B::shape(&rhs.primitive)),
        //         B::sub(lhs.primitive, rhs.primitive),
        //     ),
        //     OpsKind::UnTracked(preps) => preps.finish(B::sub(lhs.primitive, rhs.primitive)),
        // }
    }

    fn float_sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct SubScalar;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for SubScalar {
        //     type State = ();

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| grad);
        //     }
        // }

        // SubScalar
        //     .prepare([lhs.node], [lhs.graph])
        //     .stateless(B::sub_scalar(lhs.primitive, rhs))
    }

    fn float_mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Mul;

        #[derive(new, Debug)]
        struct RetroMul<B, const D: usize> {
            lhs_id: NodeID,
            rhs_id: NodeID,
            _backend: PhantomData<B>,
        }
        impl<B: Backend, const D: usize> RetroForward for RetroMul<B, D> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let lhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.lhs_id);
                let rhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.rhs_id);
                let out = B::float_mul(lhs, rhs);
                states.save(out_node, out)
            }
        }

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Mul {
            type State = (
                Option<B::FloatTensorPrimitive<D>>,
                Option<B::FloatTensorPrimitive<D>>,
                BinaryOpsBroadcast<D>,
            );

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (lhs, rhs, broadcast) = ops.state;
                // let (lhs, rhs, broadcast) = match ops.state {
                // Some(state) => state,
                // None => {
                //     let lhs: B::FloatTensorPrimitive<D> = checkpointer
                //         .retrieve_output(ops.parents[0].clone().unwrap().id.clone());
                //     let rhs: B::FloatTensorPrimitive<D> = checkpointer
                //         .retrieve_output(ops.parents[1].clone().unwrap().id.clone());
                //     let broadcast = BinaryOpsBroadcast::new::<B>(&lhs, &rhs);
                //     // TODO None if not tracked
                //     (Some(lhs), Some(rhs), broadcast)
                // }
                // };

                binary::<B, D, D, D, _, _>(
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
            .prepare([lhs.node.clone(), rhs.node.clone()], [lhs.graph, rhs.graph])
            // .compute_bound()
            .memory_bound(RetroMul::<B, D>::new(
                lhs.node.id.clone(),
                rhs.node.id.clone(),
            ))
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    rhs_tracked.then(|| lhs.primitive.clone()),
                    lhs_tracked.then(|| rhs.primitive.clone()),
                    broadcast,
                ),
                B::float_mul(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_mul(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct MulScalar;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for MulScalar {
        //     type State = FloatElem<B>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             B::mul_scalar(grad, ops.state)
        //         });
        //     }
        // }

        // match MulScalar.prepare([lhs.node], [lhs.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(rhs, B::mul_scalar(lhs.primitive, rhs)),
        //     OpsKind::UnTracked(prep) => prep.finish(B::mul_scalar(lhs.primitive, rhs)),
        // }
    }

    fn float_div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Div;
        #[derive(new, Debug)]
        struct RetroDiv<B, const D: usize> {
            lhs_parent_id: NodeID,
            rhs_parent_id: NodeID,
            _backend: PhantomData<B>,
        }

        impl<B: Backend, const D: usize> RetroForward for RetroDiv<B, D> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let lhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.lhs_parent_id);
                let rhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.rhs_parent_id);

                let out = B::float_div(lhs, rhs);
                states.save(out_node, out);
            }
        }

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Div {
            type State = (
                // CheckpointID
                Option<B::FloatTensorPrimitive<D>>,
                Option<B::FloatTensorPrimitive<D>>,
                BinaryOpsBroadcast<D>,
            );

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (lhs, rhs, broadcast) = ops.state;
                let [rhs_4lhs, rhs_4rhs] = duplicate(&ops.parents, rhs);

                binary::<B, D, D, D, _, _>(
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

        // TODO had to add a lot of clone, for nodes then for their ids. same goes for parents in backward()
        match Div
            .prepare([lhs.node.clone(), rhs.node.clone()], [lhs.graph, rhs.graph])
            // .compute_bound()
            .memory_bound(RetroDiv::<B, D>::new(
                lhs.node.id.clone(),
                rhs.node.id.clone(),
            ))
            // .state_lazy() // DELETE. Replaced by stateful where thet state is about node ids
            .stateful()
        {
            // TODO if state is lazy then we just ignore the state below. should not compute it for nothing
            OpsKind::Tracked(prep) => prep.finish(
                (
                    rhs_tracked.then(|| lhs.primitive.clone()),
                    (lhs_tracked || rhs_tracked).then(|| rhs.primitive.clone()),
                    broadcast,
                ),
                B::float_div(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_div(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct DivScalar;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for DivScalar {
        //     type State = FloatElem<B>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let tmp = 1.0 / ops.state.elem::<f32>();
        //             B::mul_scalar(grad, tmp.elem())
        //         });
        //     }
        // }

        // match DivScalar.prepare([lhs.node], [lhs.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(rhs, B::div_scalar(lhs.primitive, rhs)),
        //     OpsKind::UnTracked(prep) => prep.finish(B::div_scalar(lhs.primitive, rhs)),
        // }
    }

    fn float_matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Matmul;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Matmul {
            type State = (
                Option<B::FloatTensorPrimitive<D>>,
                Option<B::FloatTensorPrimitive<D>>,
                BinaryOpsBroadcast<D>,
            );

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (lhs, rhs, broadcast) = ops.state;

                binary::<B, D, D, D, _, _>(
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
            .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    rhs_tracked.then(|| lhs.primitive.clone()), // preps.checkpoint(lhs) -> lhs node id, keep the then
                    lhs_tracked.then(|| rhs.primitive.clone()),
                    broadcast,
                ),
                B::float_matmul(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::float_matmul(lhs.primitive, rhs.primitive)),
        }
    }

    fn float_neg<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Neg;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Neg {
        //     type State = ();

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| B::neg(grad));
        //     }
        // }

        // Neg.prepare([tensor.node], [tensor.graph])
        //     .stateless(B::neg(tensor.primitive))
    }

    fn float_recip<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Recip;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Recip {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let tensor = ops.state;
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let tmp = B::powf(tensor, -2.0);
        //             let value = B::neg(tmp);

        //             B::mul(grad, value)
        //         });
        //     }
        // }

        // match Recip.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(tensor.primitive.clone(), B::recip(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::recip(tensor.primitive)),
        // }
    }

    fn float_swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct SwapDim;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for SwapDim {
        //     type State = (usize, usize);

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let (dim1, dim2) = ops.state;

        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             B::swap_dims(grad, dim2, dim1)
        //         });
        //     }
        // }

        // let output = B::swap_dims(tensor.primitive, dim1, dim2);

        // match SwapDim.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish((dim1, dim2), output),
        //     OpsKind::UnTracked(prep) => prep.finish(output),
        // }
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        todo!()
        // #[derive(Debug)]
        // struct ReshapeDim<const D1: usize>;

        // impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D2, 1> for ReshapeDim<D1> {
        //     type State = (Shape<D1>, Shape<D2>);

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let (shape_original, shape) = ops.state;

        //         unary::<B, D2, D1, _>(ops.parents, ops.node, grads, |grad| {
        //             let shape_grad = B::shape(&grad);
        //             let mut grad = grad;

        //             for i in 0..D2 {
        //                 if shape.dims[i] == 1 && shape_grad.dims[i] != 1 {
        //                     grad = B::sum_dim(grad, i);
        //                 }
        //             }

        //             B::reshape(grad, shape_original)
        //         });
        //     }
        // }

        // match ReshapeDim.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (B::shape(&tensor.primitive), shape.clone()),
        //         B::reshape(tensor.primitive, shape),
        //     ),
        //     OpsKind::UnTracked(prep) => prep.finish(B::reshape(tensor.primitive, shape)),
        // }
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<B, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Gather;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Gather {
        //     type State = (usize, IntTensor<B, D>, Shape<D>, B::Device);

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let (dim, indices, shape, device) = ops.state;

        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let zeros = B::zeros(shape, &device);
        //             B::scatter(dim, zeros, indices, grad)
        //         });
        //     }
        // }

        // match Gather.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (
        //             dim,
        //             indices.clone(),
        //             B::shape(&tensor.primitive),
        //             B::device(&tensor.primitive),
        //         ),
        //         B::gather(dim, tensor.primitive, indices),
        //     ),
        //     OpsKind::UnTracked(prep) => prep.finish(B::gather(dim, tensor.primitive, indices)),
        // }
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<B, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Scatter;

        // impl<B: Backend, const D: usize> Backward<B, D, 2> for Scatter {
        //     type State = (usize, IntTensor<B, D>, Shape<D>, Shape<D>, B::Device);

        //     fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
        //         let (dim, indices, shape_lhs, shape_rhs, device) = ops.state;
        //         let [indices_4lhs, indices_4rhs] = duplicate(&ops.parents, Some(indices));

        //         binary::<B, D, D, D, _, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| {
        //                 let zeros = B::zeros(shape_lhs, &device);
        //                 B::scatter(dim, grad, indices_4lhs.unwrap(), zeros)
        //             },
        //             |grad| {
        //                 let zeros = B::zeros(shape_rhs, &device);
        //                 B::scatter(dim, zeros, indices_4rhs.unwrap(), grad)
        //             },
        //         );
        //     }
        // }

        // match Scatter
        //     .prepare([tensor.node, value.node], [tensor.graph, value.graph])
        //     .stateful()
        // {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (
        //             dim,
        //             indices.clone(),
        //             B::shape(&tensor.primitive),
        //             B::shape(&value.primitive),
        //             B::device(&value.primitive),
        //         ),
        //         B::scatter(dim, tensor.primitive, indices, value.primitive),
        //     ),
        //     OpsKind::UnTracked(prep) => {
        //         prep.finish(B::scatter(dim, tensor.primitive, indices, value.primitive))
        //     }
        // }
    }

    fn float_select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct IndexSelectDim;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for IndexSelectDim {
        //     type State = (usize, IntTensor<B, 1>, Shape<D>, B::Device);

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let (dim, indices, shape, device) = ops.state;

        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let zeros = B::zeros(shape, &device);
        //             B::select_assign(zeros, dim, indices, grad)
        //         });
        //     }
        // }

        // match IndexSelectDim
        //     .prepare([tensor.node], [tensor.graph])
        //     .stateful()
        // {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (
        //             dim,
        //             indices.clone(),
        //             B::shape(&tensor.primitive),
        //             B::device(&tensor.primitive),
        //         ),
        //         B::select(tensor.primitive, dim, indices),
        //     ),
        //     OpsKind::UnTracked(prep) => prep.finish(B::select(tensor.primitive, dim, indices)),
        // }
    }

    fn float_select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct IndexSelectDimAssign<const D: usize>;

        // impl<B: Backend, const D: usize> Backward<B, D, 2> for IndexSelectDimAssign<D> {
        //     type State = (usize, IntTensor<B, 1>, Shape<D>, Shape<D>, B::Device);

        //     fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
        //         let (dim, indices, shape_lhs, shape_rhs, device) = ops.state;
        //         let [indices_4lhs, indices_4rhs] = duplicate(&ops.parents, Some(indices));

        //         binary::<B, D, D, D, _, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| {
        //                 let zeros = B::zeros(shape_lhs, &device);
        //                 B::select_assign(grad, dim, indices_4lhs.unwrap(), zeros)
        //             },
        //             |grad| {
        //                 let zeros = B::zeros(shape_rhs, &device);
        //                 B::select_assign(zeros, dim, indices_4rhs.unwrap(), grad)
        //             },
        //         );
        //     }
        // }

        // match IndexSelectDimAssign::<D>
        //     .prepare([tensor.node, value.node], [tensor.graph, value.graph])
        //     .stateful()
        // {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (
        //             dim,
        //             indices.clone(),
        //             B::shape(&tensor.primitive),
        //             B::shape(&value.primitive),
        //             B::device(&value.primitive),
        //         ),
        //         B::select_assign(tensor.primitive, dim, indices, value.primitive),
        //     ),
        //     OpsKind::UnTracked(prep) => prep.finish(B::select_assign(
        //         tensor.primitive,
        //         dim,
        //         indices,
        //         value.primitive,
        //     )),
        // }
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        todo!()
        // #[derive(Debug)]
        // struct Index<const D2: usize>;

        // impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D1, 1> for Index<D2> {
        //     type State = ([std::ops::Range<usize>; D2], Shape<D1>, B::Device);

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let (ranges, shape, device) = ops.state;

        //         unary::<B, D1, D1, _>(ops.parents, ops.node, grads, |grad| {
        //             let zeros = B::zeros(shape, &device);
        //             B::slice_assign(zeros, ranges, grad)
        //         });
        //     }
        // }

        // match Index.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (
        //             ranges.clone(),
        //             B::shape(&tensor.primitive),
        //             B::device(&tensor.primitive),
        //         ),
        //         B::slice(tensor.primitive, ranges),
        //     ),
        //     OpsKind::UnTracked(prep) => prep.finish(B::slice(tensor.primitive, ranges)),
        // }
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        todo!()
        // #[derive(Debug)]
        // struct IndexAssign<const D2: usize>;

        // impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D1, 2> for IndexAssign<D2> {
        //     type State = ([std::ops::Range<usize>; D2], Shape<D1>, B::Device);

        //     fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
        //         let (ranges, shape_rhs, device) = ops.state;
        //         let [ranges_4lhs, ranges_4rhs] = duplicate(&ops.parents, Some(ranges));

        //         binary::<B, D1, D1, D1, _, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| {
        //                 let zeros = B::zeros(shape_rhs, &device);
        //                 B::slice_assign(grad, ranges_4lhs.unwrap(), zeros)
        //             },
        //             |grad| B::slice(grad, ranges_4rhs.unwrap()),
        //         );
        //     }
        // }

        // match IndexAssign
        //     .prepare([tensor.node, value.node], [tensor.graph, value.graph])
        //     .stateful()
        // {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (
        //             ranges.clone(),
        //             B::shape(&value.primitive),
        //             B::device(&value.primitive),
        //         ),
        //         B::slice_assign(tensor.primitive, ranges, value.primitive),
        //     ),
        //     OpsKind::UnTracked(prep) => {
        //         prep.finish(B::slice_assign(tensor.primitive, ranges, value.primitive))
        //     }
        // }
    }

    fn float_mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        source: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct MaskWhere;

        // impl<B: Backend, const D: usize> Backward<B, D, 2> for MaskWhere {
        //     type State = (BoolTensor<B, D>, Shape<D>, Shape<D>, B::Device);

        //     fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
        //         let (mask, shape_lhs, shape_rhs, device) = ops.state;
        //         let [mask_4lhs, mask_4rhs] = duplicate(&ops.parents, Some(mask));

        //         binary::<B, D, D, D, _, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| {
        //                 let zeros = B::zeros(shape_lhs.clone(), &device);
        //                 let grad = B::mask_where(grad, mask_4lhs.unwrap(), zeros);

        //                 broadcast_shape::<B, D>(grad, &shape_lhs)
        //             },
        //             |grad| {
        //                 let zeros = B::zeros(shape_rhs.clone(), &device);
        //                 let grad = B::mask_where(zeros, mask_4rhs.unwrap(), grad);

        //                 broadcast_shape::<B, D>(grad, &shape_rhs)
        //             },
        //         );
        //     }
        // }

        // match MaskWhere
        //     .prepare([tensor.node, source.node], [tensor.graph, source.graph])
        //     .stateful()
        // {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (
        //             mask.clone(),
        //             B::shape(&tensor.primitive),
        //             B::shape(&source.primitive),
        //             B::device(&source.primitive),
        //         ),
        //         B::mask_where(tensor.primitive, mask, source.primitive),
        //     ),
        //     OpsKind::UnTracked(prep) => {
        //         prep.finish(B::mask_where(tensor.primitive, mask, source.primitive))
        //     }
        // }
    }

    fn float_mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<B, D>,
        value: FloatElem<B>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct MaskFill;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for MaskFill {
        //     type State = BoolTensor<B, D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             B::mask_fill(grad, ops.state, 0.elem())
        //         });
        //     }
        // }

        // match MaskFill.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(mask.clone(), B::mask_fill(tensor.primitive, mask, value))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::mask_fill(tensor.primitive, mask, value)),
        // }
    }

    fn float_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::equal(lhs.primitive, rhs.primitive)
    }

    fn float_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::equal_elem(lhs.primitive, rhs)
    }

    fn float_greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::greater(lhs.primitive, rhs.primitive)
    }

    fn float_greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::greater_elem(lhs.primitive, rhs)
    }

    fn float_greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::greater_equal(lhs.primitive, rhs.primitive)
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::greater_equal_elem(lhs.primitive, rhs)
    }

    fn float_lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::lower(lhs.primitive, rhs.primitive)
    }

    fn float_lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::lower_elem(lhs.primitive, rhs)
    }

    fn float_lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::lower_equal(lhs.primitive, rhs.primitive)
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        todo!()
        // B::lower_equal_elem(lhs.primitive, rhs)
    }

    fn float_detach<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        // When we detach a tensor, we remove it from the graph, but we still want to keep the
        // `require_grad` setting.
        let is_require_grad = Self::float_is_require_grad(&tensor);
        let tensor = AutodiffTensor::new(tensor.primitive);

        match is_require_grad {
            true => tensor.require_grad(),
            false => tensor,
        }
    }

    fn float_set_require_grad<const D: usize>(
        tensor: FloatTensor<Self, D>,
        require_grad: bool,
    ) -> FloatTensor<Self, D> {
        if require_grad {
            return tensor.require_grad();
        }

        AutodiffTensor::new(tensor.primitive)
    }

    fn float_is_require_grad<const D: usize>(tensor: &FloatTensor<Self, D>) -> bool {
        matches!(tensor.node.requirement, Requirement::Grad)
    }

    fn float_mean<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        todo!()
        // #[derive(Debug)]
        // struct Mean<const D: usize>;

        // impl<B: Backend, const D: usize> Backward<B, 1, 1> for Mean<D> {
        //     type State = Shape<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, 1, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let shape = ops.state;
        //             let val = 1_f64 / shape.num_elements() as f64;
        //             let ones = B::ones(shape, &B::device(&grad));
        //             let val = B::mul_scalar(ones, val.elem());

        //             let grad: Tensor<B, 1> = Tensor::from_primitive(grad);
        //             let val: Tensor<B, D> = Tensor::from_primitive(val);

        //             val.mul(grad.unsqueeze()).into_primitive()
        //         });
        //     }
        // }

        // match Mean.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(B::shape(&tensor.primitive), B::mean(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::mean(tensor.primitive)),
        // }
    }

    fn float_sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        todo!()
        // #[derive(Debug)]
        // struct Sum<const D: usize>;

        // impl<B: Backend, const D: usize> Backward<B, 1, 1> for Sum<D> {
        //     type State = Shape<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, 1, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let val = B::ones(ops.state, &B::device(&grad));

        //             let grad: Tensor<B, 1> = Tensor::from_primitive(grad);
        //             let val: Tensor<B, D> = Tensor::from_primitive(val);

        //             val.mul(grad.unsqueeze()).into_primitive()
        //         });
        //     }
        // }

        // match Sum.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(B::shape(&tensor.primitive), B::sum(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::sum(tensor.primitive)),
        // }
    }

    fn float_mean_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct MeanDim;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for MeanDim {
        //     type State = (Shape<D>, usize);

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let (shape, dim) = ops.state;

        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let val = 1_f64 / shape.dims[dim] as f64;
        //             let ones = B::ones(shape, &B::device(&grad));
        //             let val = B::mul_scalar(ones, B::FloatElem::from_elem(val));

        //             let grad = B::sum_dim(grad, dim);
        //             B::mul(val, grad)
        //         });
        //     }
        // }

        // match MeanDim.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (B::shape(&tensor.primitive), dim),
        //         B::mean_dim(tensor.primitive, dim),
        //     ),
        //     OpsKind::UnTracked(prep) => prep.finish(B::mean_dim(tensor.primitive, dim)),
        // }
    }

    fn float_sum_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct SumDim;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for SumDim {
        //     type State = (Shape<D>, usize);

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let (shape, dim) = ops.state;

        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let ones = B::ones(shape, &B::device(&grad));
        //             let grad = B::sum_dim(grad, dim);

        //             B::mul(ones, grad)
        //         });
        //     }
        // }

        // match SumDim.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (B::shape(&tensor.primitive), dim),
        //         B::sum_dim(tensor.primitive, dim),
        //     ),
        //     OpsKind::UnTracked(prep) => prep.finish(B::sum_dim(tensor.primitive, dim)),
        // }
    }

    fn float_to_full_precision<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> FloatTensor<FullPrecisionBackend<Self>, D> {
        todo!()
        // #[derive(Debug)]
        // struct ToFullPrecision<B: Backend> {
        //     phantom: PhantomData<B>,
        // }

        // impl<B: Backend, const D: usize> Backward<B::FullPrecisionBackend, D, 1> for ToFullPrecision<B> {
        //     type State = ();

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary_different_backend::<B, B::FullPrecisionBackend, D, D, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| B::from_full_precision(grad),
        //         );
        //     }
        // }

        // let ops = ToFullPrecision::<B> {
        //     phantom: PhantomData,
        // };
        // ops.prepare([tensor.node.clone()], [tensor.graph.clone()])
        //     .stateless(B::to_full_precision(&tensor.primitive))
    }

    fn float_from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<Self>, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct FromFullPrecision<B: Backend> {
        //     phantom: PhantomData<B>,
        // }

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for FromFullPrecision<B::FullPrecisionBackend> {
        //     type State = ();

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary_different_backend::<B::FullPrecisionBackend, B, D, D, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| B::to_full_precision(&grad),
        //         );
        //     }
        // }

        // let ops = FromFullPrecision::<B::FullPrecisionBackend> {
        //     phantom: PhantomData,
        // };

        // ops.prepare([tensor.node.clone()], [tensor.graph])
        //     .stateless(B::from_full_precision(tensor.primitive))
    }

    fn float_argmax<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<B, D> {
        todo!()
        // B::argmax(tensor.primitive, dim)
    }

    fn float_argmin<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<B, D> {
        todo!()
        // B::argmin(tensor.primitive, dim)
    }

    fn float_exp<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Exp;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Exp {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| B::mul(grad, ops.state));
        //     }
        // }

        // let output = B::exp(tensor.primitive);

        // match Exp.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(output.clone(), output),
        //     OpsKind::UnTracked(prep) => prep.finish(output),
        // }
    }

    fn float_log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Log;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Log {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let value = B::powf(ops.state, -1.0);
        //             B::mul(grad, value)
        //         });
        //     }
        // }

        // match Log.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(tensor.primitive.clone(), B::log(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::log(tensor.primitive)),
        // }
    }

    fn float_log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Log1P;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Log1P {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let value = B::add_scalar(ops.state, 1.elem());
        //             let value = B::powf(value, -1.0);

        //             B::mul(grad, value)
        //         });
        //     }
        // }

        // match Log1P.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(tensor.primitive.clone(), B::log1p(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::log1p(tensor.primitive)),
        // }
    }

    fn float_powf<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct PowF;

        // impl<B: Backend, const D: usize> Backward<B, D, 2> for PowF {
        //     type State = (
        //         B::FloatTensorPrimitive<D>,
        //         B::FloatTensorPrimitive<D>,
        //         BinaryOpsBroadcast<D>,
        //     );

        //     fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
        //         let (lhs, rhs, broadcast) = ops.state;
        //         // Both lhs and rhs are needed for both lhs and rhs gradients, but we clone them
        //         // the number of times required by the parents specification.
        //         let [rhs_4lhs, rhs_4rhs] = duplicate(&ops.parents, Some(rhs));
        //         let [lhs_4lhs, lhs_4rhs] = duplicate(&ops.parents, Some(lhs));

        //         binary::<B, D, D, D, _, _>(
        //             ops.parents,
        //             ops.node,
        //             grads,
        //             |grad| {
        //                 //rhs*(lhs.val**(rhs-1))*grad
        //                 let rhs1 = rhs_4lhs.unwrap();
        //                 let rhs2 = rhs1.clone();
        //                 let lhs = lhs_4lhs.unwrap();

        //                 let tmp = B::float_powf(
        //                     lhs,
        //                     B::float_sub_scalar(rhs1, B::FloatElem::from_elem(1.0)),
        //                 );
        //                 let value = B::float_mul(tmp, rhs2);
        //                 let grad = B::float_mul(grad, value);

        //                 broadcast.backward_lhs::<B>(grad)
        //             },
        //             |grad| {
        //                 //lhs**rhs * ln(lhs) * grad
        //                 let rhs = rhs_4rhs.unwrap();
        //                 let lhs1 = lhs_4rhs.unwrap();
        //                 let lhs2 = lhs1.clone();
        //                 let tmp = B::float_powf(lhs1, rhs);
        //                 let value = B::float_mul(tmp, B::float_log(lhs2));
        //                 let grad = B::float_mul(grad, value);

        //                 broadcast.backward_rhs::<B>(grad)
        //             },
        //         );
        //     }
        //     }

        //     let broadcast = BinaryOpsBroadcast::new::<B>(&lhs.primitive, &rhs.primitive);

        //     match PowF
        //         .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
        //         .stateful()
        //     {
        //         OpsKind::Tracked(prep) => prep.finish(
        //             (lhs.primitive.clone(), rhs.primitive.clone(), broadcast),
        //             B::float_powf(lhs.primitive, rhs.primitive),
        //         ),
        //         OpsKind::UnTracked(prep) => prep.finish(B::float_powf(lhs.primitive, rhs.primitive)),
        //     }
        // }
    }

    fn float_powf_scalar<const D: usize>(
        tensor: FloatTensor<Self, D>,
        value: f32,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct PowFScalar;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for PowFScalar {
        //     type State = (B::FloatTensorPrimitive<D>, f32);

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let (tensor, value) = ops.state;

        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let tmp = B::float_powf_scalar(tensor, value - 1.0);
        //             let value = B::float_mul_scalar(tmp, value.elem());

        //             B::float_mul(grad, value)
        //         });
        //     }
        // }

        // match PowFScalar.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(
        //         (tensor.primitive.clone(), value),
        //         B::float_powf_scalar(tensor.primitive, value),
        //     ),
        //     OpsKind::UnTracked(prep) => prep.finish(B::float_powf_scalar(tensor.primitive, value)),
        // }
    }

    fn float_sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Sqrt;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Sqrt {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let input = ops.state;
        //             let value = B::div_scalar(B::powf(input, -0.5), 2.elem());

        //             B::mul(grad, value)
        //         });
        //     }
        // }

        // match Sqrt.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(tensor.primitive.clone(), B::sqrt(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::sqrt(tensor.primitive)),
        // }
    }

    fn float_abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Abs;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Abs {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| B::mul(grad, ops.state));
        //     }
        // }

        // match Abs.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let output = B::abs(tensor.primitive.clone());
        //         let state = B::div(tensor.primitive, output.clone());
        //         prep.finish(state, output)
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::abs(tensor.primitive)),
        // }
    }

    fn float_cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Cos;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Cos {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let input = ops.state;
        //             let value = B::neg(B::sin(input));

        //             B::mul(grad, value)
        //         });
        //     }
        // }

        // match Cos.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(tensor.primitive.clone(), B::cos(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::cos(tensor.primitive)),
        // }
    }

    fn float_sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Sin;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Sin {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let value = B::cos(ops.state);
        //             B::mul(grad, value)
        //         });
        //     }
        // }

        // match Sin.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(tensor.primitive.clone(), B::sin(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::sin(tensor.primitive)),
        // }
    }

    fn float_tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Tanh;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Tanh {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let value = B::add_scalar(B::neg(B::powf(ops.state, 2.0)), 1.elem());
        //             B::mul(grad, value)
        //         });
        //     }
        // }

        // match Tanh.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let output = B::tanh(tensor.primitive);
        //         prep.finish(output.clone(), output)
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::tanh(tensor.primitive)),
        // }
    }

    fn float_erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Erf;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Erf {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             let exponent = B::neg(B::powf(ops.state, 2.0));
        //             let numerator = B::mul_scalar(B::exp(exponent), 2.0.elem());
        //             let denominator = std::f64::consts::PI.sqrt().elem();
        //             let value = B::div_scalar(numerator, denominator);

        //             B::mul(grad, value)
        //         });
        //     }
        // }

        // match Erf.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         prep.finish(tensor.primitive.clone(), B::erf(tensor.primitive))
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::erf(tensor.primitive)),
        // }
    }

    fn float_cat<const D: usize>(
        tensors: Vec<FloatTensor<Self, D>>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(new, Debug)]
        // struct CatStep<B: Backend, const D: usize> {
        //     nodes: Vec<Option<NodeRef>>,
        //     // The dimension of each tensor along the dim dimension.
        //     // This indicates the number of dimension concatenated for each tensor.
        //     dim_sizes: Vec<usize>,
        //     output: NodeRef,
        //     phantom: PhantomData<B>,
        //     dim: usize,
        // }

        // impl<B: Backend, const D: usize> Step for CatStep<B, D> {
        //     fn float_step(self: Box<Self>, grads: &mut Gradients) {
        //         let grad = grads.consume::<B, D>(&self.output);
        //         let ranges: Vec<_> = B::shape(&grad).dims.iter().map(|v| 0..*v).collect();
        //         let ranges: [std::ops::Range<usize>; D] = ranges.try_into().unwrap();

        //         let mut current_index = 0;

        //         self.nodes
        //             .into_iter()
        //             .zip(self.dim_sizes)
        //             .filter_map(|(node, dim_size)| node.map(|node| (node, dim_size)))
        //             .for_each(|(node, dim_size)| {
        //                 let mut ranges = ranges.clone();
        //                 ranges[self.dim] = current_index..dim_size + current_index;
        //                 current_index += dim_size;
        //                 grads.register::<B, D>(node, B::slice(grad.clone(), ranges));
        //             });
        //     }

        //     fn float_node(&self) -> NodeRef {
        //         self.output.clone()
        //     }
        // }

        // let mut nodes = Vec::with_capacity(tensors.len());
        // let mut graphs = Vec::with_capacity(tensors.len());
        // let mut primitives = Vec::with_capacity(tensors.len());
        // let mut dim_sizes = Vec::with_capacity(tensors.len());

        // tensors.into_iter().for_each(|tensor| {
        //     dim_sizes.push(B::shape(&tensor.primitive).dims[dim]);
        //     nodes.push(tensor.node);
        //     primitives.push(tensor.primitive);
        //     graphs.push(tensor.graph);
        // });

        // let requirement = Requirement::from_nodes(&nodes);

        // let output = B::cat(primitives, dim);
        // if requirement.is_none() {
        //     return AutodiffTensor::from_parents(output, &nodes, graphs.into_iter(), requirement);
        // }

        // let output = AutodiffTensor::from_parents(output, &nodes, graphs.into_iter(), requirement);
        // let nodes = nodes
        //     .into_iter()
        //     .map(|node| node.clone_if_require_grad())
        //     .collect::<Vec<_>>();

        // let ops = CatStep::<B, D>::new(nodes, dim_sizes, output.node.clone(), dim);
        // output.register_step(ops)
    }

    fn float_max_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
        // match MaxMinDim.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let shape = B::shape(&tensor.primitive);
        //         let (tensor, index) = B::max_dim_with_indices(tensor.primitive, dim);
        //         prep.finish((index, shape), tensor)
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::max_dim(tensor.primitive, dim)),
        // }
    }
    fn float_max_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<B, D>) {
        todo!()
        // match MaxMinDim.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let shape = B::shape(&tensor.primitive);
        //         let (tensor, index) = B::max_dim_with_indices(tensor.primitive, dim);
        //         let tensor = prep.finish((index.clone(), shape), tensor);

        //         (tensor, index)
        //     }
        //     OpsKind::UnTracked(prep) => {
        //         let (tensor, index) = B::max_dim_with_indices(tensor.primitive, dim);
        //         let tensor = prep.finish(tensor);

        //         (tensor, index)
        //     }
        // }
    }
    fn float_min_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
        // match MaxMinDim.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let shape = B::shape(&tensor.primitive);
        //         let (tensor, index) = B::min_dim_with_indices(tensor.primitive, dim);
        //         prep.finish((index, shape), tensor)
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::min_dim(tensor.primitive, dim)),
        // }
    }
    fn float_min_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<B, D>) {
        todo!()
        // match MaxMinDim.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let shape = B::shape(&tensor.primitive);
        //         let (tensor, index) = B::min_dim_with_indices(tensor.primitive, dim);
        //         let tensor = prep.finish((index.clone(), shape), tensor);

        //         (tensor, index)
        //     }
        //     OpsKind::UnTracked(prep) => {
        //         let (tensor, index) = B::min_dim_with_indices(tensor.primitive, dim);
        //         let tensor = prep.finish(tensor);

        //         (tensor, index)
        //     }
        // }
    }

    fn float_into_int<const D: usize>(
        tensor: FloatTensor<Self, D>,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
        // B::into_int(tensor.primitive)
    }
}

#[derive(Debug, Clone)]
enum BinaryOpsBroadcast<const D: usize> {
    Broadcasted(Shape<D>, Shape<D>),
    None,
}

impl<const D: usize> BinaryOpsBroadcast<D> {
    fn new<B: Backend>(lhs: &B::FloatTensorPrimitive<D>, rhs: &B::FloatTensorPrimitive<D>) -> Self {
        let shape_lhs = B::float_shape(lhs);
        let shape_rhs = B::float_shape(rhs);

        for i in 0..D {
            if shape_rhs.dims[i] != shape_lhs.dims[i] {
                return Self::Broadcasted(shape_lhs, shape_rhs);
            }
        }

        Self::None
    }

    fn backward_lhs<B: Backend>(
        &self,
        grad: B::FloatTensorPrimitive<D>,
    ) -> B::FloatTensorPrimitive<D> {
        match self {
            BinaryOpsBroadcast::Broadcasted(lhs, _rhs) => broadcast_shape::<B, D>(grad, lhs),
            BinaryOpsBroadcast::None => grad,
        }
    }

    fn backward_rhs<B: Backend>(
        &self,
        grad: B::FloatTensorPrimitive<D>,
    ) -> B::FloatTensorPrimitive<D> {
        match self {
            BinaryOpsBroadcast::Broadcasted(_lhs, rhs) => broadcast_shape::<B, D>(grad, rhs),
            BinaryOpsBroadcast::None => grad,
        }
    }
}
