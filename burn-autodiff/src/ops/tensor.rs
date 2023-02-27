use crate::{
    grads::Gradients,
    ops::{Backward, OpsNodes},
    tensor::{ADTensor, BackwardTensor, BoolTensor, Elem, IntTensor},
    utils::duplicate,
    ADBackendDecorator,
};

use burn_tensor::{backend::Backend, ops::TensorOps, Data, Shape};

impl<B: Backend> TensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn from_data<const D: usize>(data: Data<Elem<B>, D>, device: &B::Device) -> ADTensor<B, D> {
        ADTensor::new(B::from_data(data, device))
    }

    fn from_data_bool<const D: usize>(data: Data<bool, D>, device: &B::Device) -> BoolTensor<B, D> {
        B::from_data_bool(data, device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: burn_tensor::Distribution<Elem<B>>,
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

    fn to_data<const D: usize>(tensor: &ADTensor<B, D>) -> Data<Elem<B>, D> {
        B::to_data(&tensor.primitive)
    }

    fn into_data<const D: usize>(tensor: ADTensor<B, D>) -> Data<Elem<B>, D> {
        B::into_data(tensor.primitive)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<B, D>) -> Shape<D> {
        B::bool_shape(tensor)
    }

    fn bool_to_data<const D: usize>(tensor: &BoolTensor<B, D>) -> Data<bool, D> {
        B::bool_to_data(tensor)
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<B, D>) -> Data<bool, D> {
        B::bool_into_data(tensor)
    }

    fn bool_into_int<const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, D> {
        B::bool_into_int(tensor)
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<B, D>,
        device: &B::Device,
    ) -> BoolTensor<B, D> {
        B::bool_to_device(tensor, device)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<B, D2> {
        B::bool_reshape(tensor, shape)
    }

    fn bool_index<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<B, D1> {
        B::bool_index(tensor, indexes)
    }

    fn device<const D: usize>(tensor: &ADTensor<B, D>) -> B::Device {
        B::device(&tensor.primitive)
    }

    fn to_device<const D: usize>(tensor: ADTensor<B, D>, device: &B::Device) -> ADTensor<B, D> {
        ADTensor::new(B::to_device(tensor.primitive, device))
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
            type State = ();

            fn backward(
                self,
                [node_lhs, node_rhs]: OpsNodes<2>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _: Self::State,
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume(&output));

                node_lhs
                    .requirements([grad_4lhs])
                    .run(|node_lhs, [grad]| grads.update(node_lhs, grad));
                node_rhs
                    .requirements([grad_4rhs])
                    .run(|node_rhs, [grad]| grads.update(node_rhs, grad));
            }
        }

        Add.run(
            (),
            B::add(lhs.primitive, rhs.primitive),
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
        )
    }

    fn add_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct AddScalar;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for AddScalar {
            type State = ();

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _: (),
            ) {
                let grad = grads.consume(&output);
                node.run(|node, _| grads.update(node, grad));
            }
        }

        AddScalar.run(
            (),
            B::add_scalar(lhs.primitive, rhs),
            [lhs.node],
            [lhs.graph],
        )
    }

    fn sub<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Sub;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Sub {
            type State = ();

            fn backward(
                self,
                [node_lhs, node_rhs]: OpsNodes<2>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _: (),
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume(&output));

                node_lhs
                    .requirements([grad_4lhs])
                    .run(|node_lhs, [grad]| grads.update(node_lhs, grad));
                node_rhs
                    .requirements([grad_4rhs])
                    .run(|node_rhs, [grad]| grads.update(node_rhs, B::neg(grad)))
            }
        }

        Sub.run(
            (),
            B::sub(lhs.primitive, rhs.primitive),
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
        )
    }

    fn sub_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct SubScalar;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for SubScalar {
            type State = ();

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _: (),
            ) {
                let grad = grads.consume(&output);
                node.run(|node, _| grads.update(node, grad));
            }
        }

        SubScalar.run(
            (),
            B::sub_scalar(lhs.primitive, rhs),
            [lhs.node],
            [lhs.graph],
        )
    }

    fn mul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Mul;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Mul {
            type State = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn backward(
                self,
                [node_lhs, node_rhs]: OpsNodes<2>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (lhs, rhs): Self::State,
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume(&output));

                node_lhs
                    .requirements([grad_4lhs, rhs])
                    .run(|node, [grad, rhs]| {
                        let grad_lhs = B::mul(grad, rhs);
                        grads.update(node, grad_lhs);
                    });

                node_rhs
                    .requirements([grad_4rhs, lhs])
                    .run(|node, [grad, lhs]| {
                        let grad_rhs = B::mul(grad, lhs);
                        grads.update(node, grad_rhs)
                    });
            }
        }

        Mul.run(
            (
                rhs.is_tracked().then(|| lhs.primitive.clone()),
                lhs.is_tracked().then(|| rhs.primitive.clone()),
            ),
            B::mul(lhs.primitive, rhs.primitive),
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
        )
    }

    fn mul_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MulScalar;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for MulScalar {
            type State = Elem<B>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                rhs: Elem<B>,
            ) {
                let grad = grads.consume(&output);

                node.run(|node, _| {
                    let grad = B::mul_scalar(grad, rhs);
                    grads.update(node, grad)
                });
            }
        }

        MulScalar.run(
            rhs,
            B::mul_scalar(lhs.primitive, rhs),
            [lhs.node],
            [lhs.graph],
        )
    }

    fn div<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Div;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Div {
            type State = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn backward(
                self,
                [node_lhs, node_rhs]: OpsNodes<2>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (lhs, rhs): Self::State,
            ) {
                let rhs = match rhs {
                    Some(rhs) => rhs,
                    None => panic!(
                        "Always needed, but set as optional when backward step is not required. {}",
                        "This avoids an uncessary clone, but if this panic, there is a bug!"
                    ),
                };

                let grad = grads.consume(&output);
                let [grad_4lhs, grad_4rhs] = duplicate([&node_lhs, &node_rhs], grad);
                let [rhs_4lhs, rhs_4rhs] = duplicate([&node_lhs, &node_rhs], rhs);

                node_lhs
                    .requirements([grad_4lhs, rhs_4lhs])
                    .run(|node, [grad, rhs]| {
                        let device = B::device(&rhs);
                        let shape = B::shape(&rhs);
                        let ones = B::ones(shape, &device);
                        let value = B::div(ones, rhs);
                        let grad = B::mul(grad, value);

                        grads.update(node, grad)
                    });

                node_rhs
                    .requirements([grad_4rhs, rhs_4rhs, lhs])
                    .run(|node, [grad, rhs, lhs]| {
                        let value = B::div(B::neg(lhs), B::powf(rhs, 2.0));
                        let grad = B::mul(grad, value);

                        grads.update(node, grad)
                    });
            }
        }

        Div.run(
            (
                rhs.is_tracked().then(|| lhs.primitive.clone()),
                (lhs.is_tracked() || rhs.is_tracked()).then(|| rhs.primitive.clone()),
            ),
            B::div(lhs.primitive, rhs.primitive),
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
        )
    }

    fn div_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct DivScalar;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for DivScalar {
            type State = (Shape<D>, B::Device, Elem<B>);

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (shape, device, rhs): Self::State,
            ) {
                let grad = grads.consume(&output);

                node.run(|node, _| {
                    let ones = B::ones(shape, &device);
                    let tmp = B::div_scalar(ones, rhs);
                    let grad = B::mul(grad, tmp);

                    grads.update(node, grad)
                });
            }
        }

        DivScalar.run(
            (B::shape(&lhs.primitive), B::device(&lhs.primitive), rhs),
            B::div_scalar(lhs.primitive, rhs),
            [lhs.node],
            [lhs.graph],
        )
    }

    fn matmul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Matmul;

        impl<B: Backend, const D: usize> Backward<B, D, 2> for Matmul {
            type State = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn backward(
                self,
                [node_lhs, node_rhs]: OpsNodes<2>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (lhs, rhs): Self::State,
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume(&output));

                node_lhs
                    .requirements([grad_4lhs, rhs])
                    .run(|node, [grad, rhs]| {
                        let rhs = B::transpose(rhs);
                        let grad_lhs = B::matmul(grad, rhs);
                        grads.update(node, grad_lhs)
                    });

                node_rhs
                    .requirements([grad_4rhs, lhs])
                    .run(|node, [grad, lhs]| {
                        let lhs = B::transpose(lhs);
                        let grad_rhs = B::matmul(lhs, grad);
                        grads.update(node, grad_rhs)
                    });
            }
        }

        Matmul.run(
            (
                rhs.is_tracked().then(|| lhs.primitive.clone()),
                lhs.is_tracked().then(|| rhs.primitive.clone()),
            ),
            B::matmul(lhs.primitive, rhs.primitive),
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
        )
    }

    fn neg<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Neg;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Neg {
            type State = ();

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (): Self::State,
            ) {
                let grad_out = grads.consume(&output);
                node.run(|node, _| grads.update(node, B::neg(grad_out)));
            }
        }

        Neg.run((), B::neg(tensor.primitive), [tensor.node], [tensor.graph])
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

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (dim1, dim2): Self::State,
            ) {
                let grad = grads.consume(&output);

                node.run(|node, _| {
                    let grad = B::swap_dims(grad, dim2, dim1);
                    grads.update(node, grad)
                });
            }
        }

        SwapDim.run(
            (dim1, dim2),
            B::swap_dims(tensor.primitive, dim1, dim2),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: ADTensor<B, D1>,
        shape: Shape<D2>,
    ) -> ADTensor<B, D2> {
        #[derive(Debug)]
        struct ReshapeDim<const D1: usize>;

        impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D2, 1> for ReshapeDim<D1> {
            type State = (Shape<D1>, Shape<D2>);

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, D2>,
                grads: &mut Gradients<B>,
                (shape_original, shape): Self::State,
            ) {
                let grad = grads.consume(&output);

                node.run(|node, _| {
                    let shape_grad = B::shape(&grad);
                    let mut grad = grad;

                    for i in 0..D2 {
                        if shape.dims[i] == 1 && shape_grad.dims[i] != 1 {
                            grad = B::sum_dim(grad, i);
                        }
                    }

                    let grad = B::reshape(grad, shape_original);
                    grads.update(node, grad)
                });
            }
        }

        ReshapeDim.run(
            (B::shape(&tensor.primitive), shape.clone()),
            B::reshape(tensor.primitive, shape),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: ADTensor<B, D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> ADTensor<B, D1> {
        #[derive(Debug)]
        struct Index<const D2: usize>;

        impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D1, 1> for Index<D2> {
            type State = ([std::ops::Range<usize>; D2], Shape<D1>, B::Device);

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, D1>,
                grads: &mut Gradients<B>,
                (indexes, shape, device): Self::State,
            ) {
                let grad = grads.consume(&output);

                node.run(|node, _| {
                    let zeros = B::zeros(shape, &device);
                    let grad = B::index_assign(zeros, indexes, grad);
                    grads.update(node, grad)
                });
            }
        }

        Index.run(
            (
                indexes.clone(),
                B::shape(&tensor.primitive),
                B::device(&tensor.primitive),
            ),
            B::index(tensor.primitive, indexes),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn index_assign<const D1: usize, const D2: usize>(
        _tensor: ADTensor<B, D1>,
        _indexes: [std::ops::Range<usize>; D2],
        _value: ADTensor<B, D1>,
    ) -> ADTensor<B, D1> {
        todo!()
    }

    fn mask_fill<const D: usize>(
        _tensor: ADTensor<B, D>,
        _mask: BoolTensor<B, D>,
        _value: Elem<B>,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn equal<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn equal_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_equal<const D: usize>(
        _lhs: ADTensor<B, D>,
        _rhs: ADTensor<B, D>,
    ) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_equal_scalar<const D: usize>(
        _lhs: ADTensor<B, D>,
        _rhs: Elem<B>,
    ) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_equal<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_equal_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn detach<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn mean<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        todo!()
    }

    fn sum<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        todo!()
    }

    fn mean_dim<const D: usize>(_tensor: ADTensor<B, D>, _dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn sum_dim<const D: usize>(_tensor: ADTensor<B, D>, _dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(
        _tensor: &ADTensor<B, D>,
    ) -> ADTensor<B::FullPrecisionBackend, D> {
        todo!()
    }

    fn from_full_precision<const D: usize>(
        _tensor: ADTensor<B::FullPrecisionBackend, D>,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn argmax<const D: usize>(_tensor: ADTensor<B, D>, _dim: usize) -> IntTensor<B, D> {
        todo!()
    }

    fn argmin<const D: usize>(_tensor: ADTensor<B, D>, _dim: usize) -> IntTensor<B, D> {
        todo!()
    }

    fn exp<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn log<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn log1p<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn powf<const D: usize>(_tensor: ADTensor<B, D>, _value: f32) -> ADTensor<B, D> {
        todo!()
    }

    fn sqrt<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn cos<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn sin<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn tanh<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn erf<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn cat<const D: usize>(_tensors: Vec<ADTensor<B, D>>, _dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn relu<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }
}
