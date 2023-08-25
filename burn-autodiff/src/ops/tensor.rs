use std::marker::PhantomData;

use crate::{
    grads::Gradients,
    graph::{NodeRef, Requirement, Step},
    ops::{binary, unary, unary_different_backend, Backward, Ops, OpsKind},
    tensor::{ADTensor, BoolTensor, FloatElem, IntTensor},
    utils::duplicate,
    ADBackendDecorator,
};

use burn_tensor::{backend::Backend, ops::TensorOps, Data, ElementConversion, Shape, Tensor};

use super::maxmin::MaxMinDim;

impl<B: Backend> TensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn from_data<const D: usize>(
        data: Data<FloatElem<B>, D>,
        device: &B::Device,
    ) -> ADTensor<B, D> {
        ADTensor::new(B::from_data(data, device))
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: burn_tensor::Distribution<FloatElem<B>>,
        device: &B::Device,
    ) -> ADTensor<B, D> {
        ADTensor::new(B::random(shape, distribution, device))
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        Self::from_data(Data::zeros(shape), device)
    }

    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        Self::from_data(Data::ones(shape), device)
    }

    fn shape<const D: usize>(tensor: &ADTensor<B, D>) -> Shape<D> {
        B::shape(&tensor.primitive)
    }

    fn to_data<const D: usize>(tensor: &ADTensor<B, D>) -> Data<FloatElem<B>, D> {
        B::to_data(&tensor.primitive)
    }

    fn into_data<const D: usize>(tensor: ADTensor<B, D>) -> Data<FloatElem<B>, D> {
        B::into_data(tensor.primitive)
    }

    fn device<const D: usize>(tensor: &ADTensor<B, D>) -> B::Device {
        B::device(&tensor.primitive)
    }

    fn to_device<const D: usize>(tensor: ADTensor<B, D>, device: &B::Device) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct ToDevice;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for ToDevice {
            type State = B::Device;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::to_device(grad, &ops.state)
                });
            }
        }

        match ToDevice.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let device_old = B::device(&tensor.primitive);
                prep.finish(device_old, B::to_device(tensor.primitive, device))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::to_device(tensor.primitive, device)),
        }
    }

    fn arange(range: std::ops::Range<usize>, device: &B::Device) -> IntTensor<B, 1> {
        B::arange(range, device)
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        ADTensor::new(B::empty(shape, device))
    }

    fn add<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Add;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Add {
            type State = (Shape<D>, Shape<D>);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (shape_lhs, shape_rhs) = ops.state;

                binary::<B, D, D, D, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| broadcast_shape::<B, D>(grad, shape_lhs),
                    |grad| broadcast_shape::<B, D>(grad, shape_rhs),
                );
            }
        }

        match Add
            .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
            .statefull()
        {
            OpsKind::Tracked(preps) => preps.finish(
                (B::shape(&lhs.primitive), B::shape(&rhs.primitive)),
                B::add(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(preps) => preps.finish(B::add(lhs.primitive, rhs.primitive)),
        }
    }

    fn add_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: FloatElem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct AddScalar;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for AddScalar {
            type State = ();

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| grad);
            }
        }

        AddScalar
            .prepare([lhs.node], [lhs.graph])
            .stateless(B::add_scalar(lhs.primitive, rhs))
    }

    fn sub<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Sub;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Sub {
            type State = (Shape<D>, Shape<D>);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (shape_lhs, shape_rhs) = ops.state;

                binary::<B, D, D, D, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| broadcast_shape::<B, D>(grad, shape_lhs),
                    |grad| broadcast_shape::<B, D>(B::neg(grad), shape_rhs),
                );
            }
        }

        match Sub
            .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
            .statefull()
        {
            OpsKind::Tracked(preps) => preps.finish(
                (B::shape(&lhs.primitive), B::shape(&rhs.primitive)),
                B::sub(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(preps) => preps.finish(B::sub(lhs.primitive, rhs.primitive)),
        }
    }

    fn sub_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: FloatElem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct SubScalar;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for SubScalar {
            type State = ();

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| grad);
            }
        }

        SubScalar
            .prepare([lhs.node], [lhs.graph])
            .stateless(B::sub_scalar(lhs.primitive, rhs))
    }

    fn mul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Mul;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Mul {
            type State = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (lhs, rhs) = ops.state;

                binary::<B, D, D, D, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| B::mul(grad, rhs.unwrap()),
                    |grad| B::mul(grad, lhs.unwrap()),
                );
            }
        }

        let lhs_tracked = lhs.is_tracked();
        let rhs_tracked = rhs.is_tracked();

        match Mul
            .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    rhs_tracked.then(|| lhs.primitive.clone()),
                    lhs_tracked.then(|| rhs.primitive.clone()),
                ),
                B::mul(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::mul(lhs.primitive, rhs.primitive)),
        }
    }

    fn mul_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: FloatElem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MulScalar;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for MulScalar {
            type State = FloatElem<B>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::mul_scalar(grad, ops.state)
                });
            }
        }

        match MulScalar.prepare([lhs.node], [lhs.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(rhs, B::mul_scalar(lhs.primitive, rhs)),
            OpsKind::UnTracked(prep) => prep.finish(B::mul_scalar(lhs.primitive, rhs)),
        }
    }

    fn div<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Div;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Div {
            type State = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (lhs, rhs) = ops.state;
                let [rhs_4lhs, rhs_4rhs] = duplicate(&ops.parents, rhs);

                binary::<B, D, D, D, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let rhs = rhs_4lhs.unwrap();
                        let value = B::powf(rhs, -1.0);

                        B::mul(grad, value)
                    },
                    |grad| {
                        let rhs = rhs_4rhs.unwrap();
                        let lhs = lhs.unwrap();
                        let value = B::div(B::neg(lhs), B::powf(rhs, 2.0));
                        B::mul(grad, value)
                    },
                );
            }
        }

        let lhs_tracked = lhs.is_tracked();
        let rhs_tracked = rhs.is_tracked();

        match Div
            .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    rhs_tracked.then(|| lhs.primitive.clone()),
                    (lhs_tracked || rhs_tracked).then(|| rhs.primitive.clone()),
                ),
                B::div(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::div(lhs.primitive, rhs.primitive)),
        }
    }

    fn div_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: FloatElem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct DivScalar;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for DivScalar {
            type State = FloatElem<B>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let tmp = 1.0 / ops.state.elem::<f32>();
                    B::mul_scalar(grad, tmp.elem())
                });
            }
        }

        match DivScalar.prepare([lhs.node], [lhs.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(rhs, B::div_scalar(lhs.primitive, rhs)),
            OpsKind::UnTracked(prep) => prep.finish(B::div_scalar(lhs.primitive, rhs)),
        }
    }

    fn matmul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Matmul;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Matmul {
            type State = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (lhs, rhs) = ops.state;

                binary::<B, D, D, D, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let rhs = B::transpose(rhs.unwrap());
                        B::matmul(grad, rhs)
                    },
                    |grad| {
                        let lhs = B::transpose(lhs.unwrap());
                        B::matmul(lhs, grad)
                    },
                );
            }
        }

        let lhs_tracked = lhs.is_tracked();
        let rhs_tracked = rhs.is_tracked();

        match Matmul
            .prepare([lhs.node, rhs.node], [lhs.graph, rhs.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    rhs_tracked.then(|| lhs.primitive.clone()),
                    lhs_tracked.then(|| rhs.primitive.clone()),
                ),
                B::matmul(lhs.primitive, rhs.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::matmul(lhs.primitive, rhs.primitive)),
        }
    }

    fn neg<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Neg;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Neg {
            type State = ();

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| B::neg(grad));
            }
        }

        Neg.prepare([tensor.node], [tensor.graph])
            .stateless(B::neg(tensor.primitive))
    }

    fn swap_dims<const D: usize>(
        tensor: ADTensor<B, D>,
        dim1: usize,
        dim2: usize,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct SwapDim;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for SwapDim {
            type State = (usize, usize);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (dim1, dim2) = ops.state;

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::swap_dims(grad, dim2, dim1)
                });
            }
        }

        let output = B::swap_dims(tensor.primitive, dim1, dim2);

        match SwapDim.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish((dim1, dim2), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: ADTensor<B, D1>,
        shape: Shape<D2>,
    ) -> ADTensor<B, D2> {
        #[derive(Debug)]
        struct ReshapeDim<const D1: usize>;

        impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D2, 1> for ReshapeDim<D1> {
            type State = (Shape<D1>, Shape<D2>);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (shape_original, shape) = ops.state;

                unary::<B, D2, D1, _>(ops.parents, ops.node, grads, |grad| {
                    let shape_grad = B::shape(&grad);
                    let mut grad = grad;

                    for i in 0..D2 {
                        if shape.dims[i] == 1 && shape_grad.dims[i] != 1 {
                            grad = B::sum_dim(grad, i);
                        }
                    }

                    B::reshape(grad, shape_original)
                });
            }
        }

        match ReshapeDim
            .prepare([tensor.node], [tensor.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (B::shape(&tensor.primitive), shape.clone()),
                B::reshape(tensor.primitive, shape),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::reshape(tensor.primitive, shape)),
        }
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: ADTensor<B, D>,
        indices: IntTensor<B, D>,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Gather;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Gather {
            type State = (usize, IntTensor<B, D>, Shape<D>, B::Device);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (dim, indices, shape, device) = ops.state;

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let zeros = B::zeros(shape, &device);
                    B::scatter(dim, zeros, indices, grad)
                });
            }
        }

        match Gather.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    dim,
                    indices.clone(),
                    B::shape(&tensor.primitive),
                    B::device(&tensor.primitive),
                ),
                B::gather(dim, tensor.primitive, indices),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::gather(dim, tensor.primitive, indices)),
        }
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: ADTensor<B, D>,
        indices: IntTensor<B, D>,
        value: ADTensor<B, D>,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Scatter;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Scatter {
            type State = (usize, IntTensor<B, D>, Shape<D>, Shape<D>, B::Device);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (dim, indices, shape_lhs, shape_rhs, device) = ops.state;
                let [indices_4lhs, indices_4rhs] = duplicate(&ops.parents, Some(indices));

                binary::<B, D, D, D, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let zeros = B::zeros(shape_lhs, &device);
                        B::scatter(dim, grad, indices_4lhs.unwrap(), zeros)
                    },
                    |grad| {
                        let zeros = B::zeros(shape_rhs, &device);
                        B::scatter(dim, zeros, indices_4rhs.unwrap(), grad)
                    },
                );
            }
        }

        match Scatter
            .prepare([tensor.node, value.node], [tensor.graph, value.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    dim,
                    indices.clone(),
                    B::shape(&tensor.primitive),
                    B::shape(&value.primitive),
                    B::device(&value.primitive),
                ),
                B::scatter(dim, tensor.primitive, indices, value.primitive),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::scatter(dim, tensor.primitive, indices, value.primitive))
            }
        }
    }

    fn select<const D: usize>(
        tensor: ADTensor<B, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct IndexSelectDim;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for IndexSelectDim {
            type State = (usize, IntTensor<B, 1>, Shape<D>, B::Device);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (dim, indices, shape, device) = ops.state;

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let zeros = B::zeros(shape, &device);
                    B::select_assign(zeros, dim, indices, grad)
                });
            }
        }

        match IndexSelectDim
            .prepare([tensor.node], [tensor.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    dim,
                    indices.clone(),
                    B::shape(&tensor.primitive),
                    B::device(&tensor.primitive),
                ),
                B::select(tensor.primitive, dim, indices),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::select(tensor.primitive, dim, indices)),
        }
    }

    fn select_assign<const D: usize>(
        tensor: ADTensor<B, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
        value: ADTensor<B, D>,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct IndexSelectDimAssign<const D: usize>;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for IndexSelectDimAssign<D> {
            type State = (usize, IntTensor<B, 1>, Shape<D>, Shape<D>, B::Device);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (dim, indices, shape_lhs, shape_rhs, device) = ops.state;
                let [indices_4lhs, indices_4rhs] = duplicate(&ops.parents, Some(indices));

                binary::<B, D, D, D, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let zeros = B::zeros(shape_lhs, &device);
                        B::select_assign(grad, dim, indices_4lhs.unwrap(), zeros)
                    },
                    |grad| {
                        let zeros = B::zeros(shape_rhs, &device);
                        B::select_assign(zeros, dim, indices_4rhs.unwrap(), grad)
                    },
                );
            }
        }

        match IndexSelectDimAssign::<D>
            .prepare([tensor.node, value.node], [tensor.graph, value.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    dim,
                    indices.clone(),
                    B::shape(&tensor.primitive),
                    B::shape(&value.primitive),
                    B::device(&value.primitive),
                ),
                B::select_assign(tensor.primitive, dim, indices, value.primitive),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::select_assign(
                tensor.primitive,
                dim,
                indices,
                value.primitive,
            )),
        }
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: ADTensor<B, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> ADTensor<B, D1> {
        #[derive(Debug)]
        struct Index<const D2: usize>;

        impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D1, 1> for Index<D2> {
            type State = ([std::ops::Range<usize>; D2], Shape<D1>, B::Device);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (ranges, shape, device) = ops.state;

                unary::<B, D1, D1, _>(ops.parents, ops.node, grads, |grad| {
                    let zeros = B::zeros(shape, &device);
                    B::slice_assign(zeros, ranges, grad)
                });
            }
        }

        match Index.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    ranges.clone(),
                    B::shape(&tensor.primitive),
                    B::device(&tensor.primitive),
                ),
                B::slice(tensor.primitive, ranges),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::slice(tensor.primitive, ranges)),
        }
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: ADTensor<B, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: ADTensor<B, D1>,
    ) -> ADTensor<B, D1> {
        #[derive(Debug)]
        struct IndexAssign<const D2: usize>;

        impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D1, 2> for IndexAssign<D2> {
            type State = ([std::ops::Range<usize>; D2], Shape<D1>, B::Device);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (ranges, shape_rhs, device) = ops.state;
                let [ranges_4lhs, ranges_4rhs] = duplicate(&ops.parents, Some(ranges));

                binary::<B, D1, D1, D1, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let zeros = B::zeros(shape_rhs, &device);
                        B::slice_assign(grad, ranges_4lhs.unwrap(), zeros)
                    },
                    |grad| B::slice(grad, ranges_4rhs.unwrap()),
                );
            }
        }

        match IndexAssign
            .prepare([tensor.node, value.node], [tensor.graph, value.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    ranges.clone(),
                    B::shape(&value.primitive),
                    B::device(&value.primitive),
                ),
                B::slice_assign(tensor.primitive, ranges, value.primitive),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::slice_assign(tensor.primitive, ranges, value.primitive))
            }
        }
    }

    fn mask_where<const D: usize>(
        tensor: ADTensor<B, D>,
        mask: BoolTensor<B, D>,
        source: ADTensor<B, D>,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MaskWhere;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for MaskWhere {
            type State = (BoolTensor<B, D>, Shape<D>, Shape<D>, B::Device);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let (mask, shape_lhs, shape_rhs, device) = ops.state;
                let [mask_4lhs, mask_4rhs] = duplicate(&ops.parents, Some(mask));

                binary::<B, D, D, D, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| {
                        let zeros = B::zeros(shape_lhs, &device);
                        B::mask_where(grad, mask_4lhs.unwrap(), zeros)
                    },
                    |grad| {
                        let zeros = B::zeros(shape_rhs, &device);
                        B::mask_where(zeros, mask_4rhs.unwrap(), grad)
                    },
                );
            }
        }

        match MaskWhere
            .prepare([tensor.node, source.node], [tensor.graph, source.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    mask.clone(),
                    B::shape(&tensor.primitive),
                    B::shape(&source.primitive),
                    B::device(&source.primitive),
                ),
                B::mask_where(tensor.primitive, mask, source.primitive),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::mask_where(tensor.primitive, mask, source.primitive))
            }
        }
    }

    fn mask_fill<const D: usize>(
        tensor: ADTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: FloatElem<B>,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MaskFill;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for MaskFill {
            type State = BoolTensor<B, D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::mask_fill(grad, ops.state, 0.elem())
                });
            }
        }

        match MaskFill.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(mask.clone(), B::mask_fill(tensor.primitive, mask, value))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::mask_fill(tensor.primitive, mask, value)),
        }
    }

    fn equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::equal(lhs.primitive, rhs.primitive)
    }

    fn equal_elem<const D: usize>(lhs: ADTensor<B, D>, rhs: FloatElem<B>) -> BoolTensor<B, D> {
        B::equal_elem(lhs.primitive, rhs)
    }

    fn greater<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::greater(lhs.primitive, rhs.primitive)
    }

    fn greater_elem<const D: usize>(lhs: ADTensor<B, D>, rhs: FloatElem<B>) -> BoolTensor<B, D> {
        B::greater_elem(lhs.primitive, rhs)
    }

    fn greater_equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::greater_equal(lhs.primitive, rhs.primitive)
    }

    fn greater_equal_elem<const D: usize>(
        lhs: ADTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        B::greater_equal_elem(lhs.primitive, rhs)
    }

    fn lower<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::lower(lhs.primitive, rhs.primitive)
    }

    fn lower_elem<const D: usize>(lhs: ADTensor<B, D>, rhs: FloatElem<B>) -> BoolTensor<B, D> {
        B::lower_elem(lhs.primitive, rhs)
    }

    fn lower_equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::lower_equal(lhs.primitive, rhs.primitive)
    }

    fn lower_equal_elem<const D: usize>(
        lhs: ADTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        B::lower_equal_elem(lhs.primitive, rhs)
    }

    fn detach<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        // When we detach a tensor, we remove it from the graph, but we still want to keep the
        // `require_grad` setting.
        let is_require_grad = Self::is_require_grad(&tensor);
        let tensor = ADTensor::new(tensor.primitive);

        match is_require_grad {
            true => tensor.require_grad(),
            false => tensor,
        }
    }

    fn set_require_grad<const D: usize>(
        tensor: ADTensor<B, D>,
        require_grad: bool,
    ) -> ADTensor<B, D> {
        if require_grad {
            return tensor.require_grad();
        }

        ADTensor::new(tensor.primitive)
    }

    fn is_require_grad<const D: usize>(tensor: &ADTensor<B, D>) -> bool {
        matches!(tensor.node.requirement, Requirement::Grad)
    }

    fn mean<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        #[derive(Debug)]
        struct Mean<const D: usize>;

        impl<B: Backend, const D: usize> Backward<B, 1, 1> for Mean<D> {
            type State = Shape<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, 1, D, _>(ops.parents, ops.node, grads, |grad| {
                    let shape = ops.state;
                    let val = 1_f64 / shape.num_elements() as f64;
                    let ones = B::ones(shape, &B::device(&grad));
                    let val = B::mul_scalar(ones, val.elem());

                    let grad: Tensor<B, 1> = Tensor::from_primitive(grad);
                    let val: Tensor<B, D> = Tensor::from_primitive(val);

                    val.mul(grad.unsqueeze()).into_primitive()
                });
            }
        }

        match Mean.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(B::shape(&tensor.primitive), B::mean(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::mean(tensor.primitive)),
        }
    }

    fn sum<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        #[derive(Debug)]
        struct Sum<const D: usize>;

        impl<B: Backend, const D: usize> Backward<B, 1, 1> for Sum<D> {
            type State = Shape<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, 1, D, _>(ops.parents, ops.node, grads, |grad| {
                    let val = B::ones(ops.state, &B::device(&grad));

                    let grad: Tensor<B, 1> = Tensor::from_primitive(grad);
                    let val: Tensor<B, D> = Tensor::from_primitive(val);

                    val.mul(grad.unsqueeze()).into_primitive()
                });
            }
        }

        match Sum.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(B::shape(&tensor.primitive), B::sum(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::sum(tensor.primitive)),
        }
    }

    fn mean_dim<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MeamDim;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for MeamDim {
            type State = (Shape<D>, usize);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (shape, dim) = ops.state;

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let val = 1_f64 / shape.dims[dim] as f64;
                    let ones = B::ones(shape, &B::device(&grad));
                    let val = B::mul_scalar(ones, B::FloatElem::from_elem(val));

                    let grad = B::sum_dim(grad, dim);
                    B::mul(val, grad)
                });
            }
        }

        match MeamDim.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(
                (B::shape(&tensor.primitive), dim),
                B::mean_dim(tensor.primitive, dim),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::mean_dim(tensor.primitive, dim)),
        }
    }

    fn sum_dim<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct SumDim;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for SumDim {
            type State = (Shape<D>, usize);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (shape, dim) = ops.state;

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let ones = B::ones(shape, &B::device(&grad));
                    let grad = B::sum_dim(grad, dim);

                    B::mul(ones, grad)
                });
            }
        }

        match SumDim.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(
                (B::shape(&tensor.primitive), dim),
                B::sum_dim(tensor.primitive, dim),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::sum_dim(tensor.primitive, dim)),
        }
    }

    fn to_full_precision<const D: usize>(
        tensor: &ADTensor<B, D>,
    ) -> ADTensor<B::FullPrecisionBackend, D> {
        #[derive(Debug)]
        struct ToFullPrecision<B: Backend> {
            phantom: PhantomData<B>,
        }

        impl<B: Backend, const D: usize> Backward<B::FullPrecisionBackend, D, 1> for ToFullPrecision<B> {
            type State = ();

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary_different_backend::<B, B::FullPrecisionBackend, D, D, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| B::from_full_precision(grad),
                );
            }
        }

        let ops = ToFullPrecision::<B> {
            phantom: PhantomData,
        };
        ops.prepare([tensor.node.clone()], [tensor.graph.clone()])
            .stateless(B::to_full_precision(&tensor.primitive))
    }

    fn from_full_precision<const D: usize>(
        tensor: ADTensor<B::FullPrecisionBackend, D>,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct FromFullPrecision<B: Backend> {
            phantom: PhantomData<B>,
        }

        impl<B: Backend, const D: usize> Backward<B, D, 1> for FromFullPrecision<B::FullPrecisionBackend> {
            type State = ();

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary_different_backend::<B::FullPrecisionBackend, B, D, D, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| B::to_full_precision(&grad),
                );
            }
        }

        let ops = FromFullPrecision::<B::FullPrecisionBackend> {
            phantom: PhantomData,
        };

        ops.prepare([tensor.node.clone()], [tensor.graph])
            .stateless(B::from_full_precision(tensor.primitive))
    }

    fn argmax<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> IntTensor<B, D> {
        B::argmax(tensor.primitive, dim)
    }

    fn argmin<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> IntTensor<B, D> {
        B::argmin(tensor.primitive, dim)
    }

    fn exp<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Exp;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Exp {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| B::mul(grad, ops.state));
            }
        }

        let output = B::exp(tensor.primitive);

        match Exp.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(output.clone(), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }

    fn log<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Log;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Log {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::powf(ops.state, -1.0);
                    B::mul(grad, value)
                });
            }
        }

        match Log.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(tensor.primitive.clone(), B::log(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::log(tensor.primitive)),
        }
    }

    fn log1p<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Log1P;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Log1P {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::add_scalar(ops.state, 1.elem());
                    let value = B::powf(value, -1.0);

                    B::mul(grad, value)
                });
            }
        }

        match Log1P.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(tensor.primitive.clone(), B::log1p(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::log1p(tensor.primitive)),
        }
    }

    fn powf<const D: usize>(tensor: ADTensor<B, D>, value: f32) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct PowF;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for PowF {
            type State = (B::TensorPrimitive<D>, f32);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (tensor, value) = ops.state;

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let tmp = B::powf(tensor, value - 1.0);
                    let value = B::mul_scalar(tmp, value.elem());

                    B::mul(grad, value)
                });
            }
        }

        match PowF.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(
                (tensor.primitive.clone(), value),
                B::powf(tensor.primitive, value),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::powf(tensor.primitive, value)),
        }
    }

    fn sqrt<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Sqrt;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Sqrt {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let input = ops.state;
                    let value = B::div_scalar(B::powf(input, -0.5), 2.elem());

                    B::mul(grad, value)
                });
            }
        }

        match Sqrt.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(tensor.primitive.clone(), B::sqrt(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::sqrt(tensor.primitive)),
        }
    }

    fn abs<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Abs;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Abs {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| B::mul(grad, ops.state));
            }
        }

        match Abs.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::abs(tensor.primitive.clone());
                let state = B::div(tensor.primitive, output.clone());
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::abs(tensor.primitive)),
        }
    }

    fn cos<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Cos;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Cos {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let input = ops.state;
                    let value = B::neg(B::sin(input));

                    B::mul(grad, value)
                });
            }
        }

        match Cos.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(tensor.primitive.clone(), B::cos(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::cos(tensor.primitive)),
        }
    }

    fn sin<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Sin;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Sin {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::cos(ops.state);
                    B::mul(grad, value)
                });
            }
        }

        match Sin.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(tensor.primitive.clone(), B::sin(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::sin(tensor.primitive)),
        }
    }

    fn tanh<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Tanh;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Tanh {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let value = B::add_scalar(B::neg(B::powf(ops.state, 2.0)), 1.elem());
                    B::mul(grad, value)
                });
            }
        }

        match Tanh.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::tanh(tensor.primitive);
                prep.finish(output.clone(), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::tanh(tensor.primitive)),
        }
    }

    fn erf<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Erf;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Erf {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    let exponent = B::neg(B::powf(ops.state, 2.0));
                    let numerator = B::mul_scalar(B::exp(exponent), 2.0.elem());
                    let denominator = std::f64::consts::PI.sqrt().elem();
                    let value = B::div_scalar(numerator, denominator);

                    B::mul(grad, value)
                });
            }
        }

        match Erf.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                prep.finish(tensor.primitive.clone(), B::erf(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::erf(tensor.primitive)),
        }
    }

    fn cat<const D: usize>(tensors: Vec<ADTensor<B, D>>, dim: usize) -> ADTensor<B, D> {
        #[derive(new, Debug)]
        struct CatStep<B: Backend, const D: usize> {
            nodes: Vec<Option<NodeRef>>,
            // The dimension of each tensor along the dim dimension.
            // This indicates the number of dimension concatenated for each tensor.
            dim_sizes: Vec<usize>,
            output: NodeRef,
            phantom: PhantomData<B>,
            dim: usize,
        }

        impl<B: Backend, const D: usize> Step for CatStep<B, D> {
            fn step(self: Box<Self>, grads: &mut Gradients) {
                let grad = grads.consume::<B, D>(&self.output);
                let ranges: Vec<_> = B::shape(&grad).dims.iter().map(|v| 0..*v).collect();
                let ranges: [std::ops::Range<usize>; D] = ranges.try_into().unwrap();

                let mut current_index = 0;

                self.nodes
                    .into_iter()
                    .zip(self.dim_sizes)
                    .filter_map(|(node, dim_size)| node.map(|node| (node, dim_size)))
                    .for_each(|(node, dim_size)| {
                        let mut ranges = ranges.clone();
                        ranges[self.dim] = current_index..dim_size + current_index;
                        current_index += dim_size;
                        grads.register::<B, D>(node, B::slice(grad.clone(), ranges));
                    });
            }

            fn node(&self) -> NodeRef {
                self.output.clone()
            }
        }

        let mut nodes = Vec::with_capacity(tensors.len());
        let mut graphs = Vec::with_capacity(tensors.len());
        let mut primitives = Vec::with_capacity(tensors.len());
        let mut dim_sizes = Vec::with_capacity(tensors.len());

        tensors.into_iter().for_each(|tensor| {
            dim_sizes.push(B::shape(&tensor.primitive).dims[dim]);
            nodes.push(tensor.node);
            primitives.push(tensor.primitive);
            graphs.push(tensor.graph);
        });

        let requirement = Requirement::from_nodes(&nodes);

        let output = B::cat(primitives, dim);
        if requirement.is_none() {
            return ADTensor::from_parents(output, &nodes, graphs.into_iter(), requirement);
        }

        let output = ADTensor::from_parents(output, &nodes, graphs.into_iter(), requirement);
        let nodes = nodes
            .into_iter()
            .map(|node| node.clone_if_require_grad())
            .collect::<Vec<_>>();

        let ops = CatStep::<B, D>::new(nodes, dim_sizes, output.node.clone(), dim);
        output.register_step(ops)
    }

    fn max_dim<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> ADTensor<B, D> {
        match MaxMinDim.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let shape = B::shape(&tensor.primitive);
                let (tensor, index) = B::max_dim_with_indices(tensor.primitive, dim);
                prep.finish((index, shape), tensor)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::max_dim(tensor.primitive, dim)),
        }
    }
    fn max_dim_with_indices<const D: usize>(
        tensor: ADTensor<B, D>,
        dim: usize,
    ) -> (ADTensor<B, D>, IntTensor<B, D>) {
        match MaxMinDim.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let shape = B::shape(&tensor.primitive);
                let (tensor, index) = B::max_dim_with_indices(tensor.primitive, dim);
                let tensor = prep.finish((index.clone(), shape), tensor);

                (tensor, index)
            }
            OpsKind::UnTracked(prep) => {
                let (tensor, index) = B::max_dim_with_indices(tensor.primitive, dim);
                let tensor = prep.finish(tensor);

                (tensor, index)
            }
        }
    }
    fn min_dim<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> ADTensor<B, D> {
        match MaxMinDim.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let shape = B::shape(&tensor.primitive);
                let (tensor, index) = B::min_dim_with_indices(tensor.primitive, dim);
                prep.finish((index, shape), tensor)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::min_dim(tensor.primitive, dim)),
        }
    }
    fn min_dim_with_indices<const D: usize>(
        tensor: ADTensor<B, D>,
        dim: usize,
    ) -> (ADTensor<B, D>, IntTensor<B, D>) {
        match MaxMinDim.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let shape = B::shape(&tensor.primitive);
                let (tensor, index) = B::min_dim_with_indices(tensor.primitive, dim);
                let tensor = prep.finish((index.clone(), shape), tensor);

                (tensor, index)
            }
            OpsKind::UnTracked(prep) => {
                let (tensor, index) = B::min_dim_with_indices(tensor.primitive, dim);
                let tensor = prep.finish(tensor);

                (tensor, index)
            }
        }
    }

    fn into_int<const D: usize>(
        tensor: ADTensor<B, D>,
    ) -> <ADBackendDecorator<B> as Backend>::IntTensorPrimitive<D> {
        B::into_int(tensor.primitive)
    }
}

/// Make sure the grad tensor has the given shape.
///
/// If broadcasting happened during the forward pass, the gradients will be sum along the
/// broadcasted dimension.
fn broadcast_shape<B: Backend, const D: usize>(
    mut grad: B::TensorPrimitive<D>,
    shape: Shape<D>,
) -> B::TensorPrimitive<D> {
    let shape_grad = B::shape(&grad);

    for i in 0..D {
        if shape_grad.dims[i] != shape.dims[i] {
            if shape.dims[i] != 1 {
                panic!(
                    "Invalid broadcast shapes: Next grad shape {:?}, Previous grad shape {:?}. {}",
                    shape.dims, shape_grad.dims, "Expected the shape of the next grad to be 1."
                );
            }
            grad = B::sum_dim(grad, i);
        }
    }

    grad
}
