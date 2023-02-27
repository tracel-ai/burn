use std::marker::PhantomData;

use crate::{
    grads::Gradients,
    graph::{NodeRef, Requirement, Step},
    ops::{Backward, OpsNode, OpsNodes},
    tensor::{ADTensor, BoolTensor, Elem, IntTensor},
    utils::duplicate,
    ADBackendDecorator,
};

use burn_tensor::{backend::Backend, ops::TensorOps, Data, ElementConversion, Shape, Tensor};

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
                node_output: NodeRef,
                grads: &mut Gradients,
                _: Self::State,
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume::<B, D>(&node_output));

                node_lhs
                    .requirements([grad_4lhs])
                    .run(|node_lhs, [grad]| grads.register::<B, D>(node_lhs, grad));
                node_rhs
                    .requirements([grad_4rhs])
                    .run(|node_rhs, [grad]| grads.register::<B, D>(node_rhs, grad));
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
                node_output: NodeRef,
                grads: &mut Gradients,
                _: (),
            ) {
                let grad = grads.consume::<B, D>(&node_output);
                node.run(|node, _| grads.register::<B, D>(node, grad));
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
                node_output: NodeRef,
                grads: &mut Gradients,
                _: (),
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume::<B, D>(&node_output));

                node_lhs
                    .requirements([grad_4lhs])
                    .run(|node_lhs, [grad]| grads.register::<B, D>(node_lhs, grad));
                node_rhs
                    .requirements([grad_4rhs])
                    .run(|node_rhs, [grad]| grads.register::<B, D>(node_rhs, B::neg(grad)))
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
                node_output: NodeRef,
                grads: &mut Gradients,
                _: (),
            ) {
                let grad = grads.consume::<B, D>(&node_output);
                node.run(|node, _| grads.register::<B, D>(node, grad));
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
                node_output: NodeRef,
                grads: &mut Gradients,
                (lhs, rhs): Self::State,
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume::<B, D>(&node_output));

                node_lhs
                    .requirements([grad_4lhs, rhs])
                    .run(|node, [grad, rhs]| {
                        let grad_lhs = B::mul(grad, rhs);
                        grads.register::<B, D>(node, grad_lhs);
                    });

                node_rhs
                    .requirements([grad_4rhs, lhs])
                    .run(|node, [grad, lhs]| {
                        let grad_rhs = B::mul(grad, lhs);
                        grads.register::<B, D>(node, grad_rhs)
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
                node_output: NodeRef,
                grads: &mut Gradients,
                rhs: Elem<B>,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.run(|node, _| {
                    let grad = B::mul_scalar(grad, rhs);
                    grads.register::<B, D>(node, grad)
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
                node_output: NodeRef,
                grads: &mut Gradients,
                (lhs, rhs): Self::State,
            ) {
                let rhs = match rhs {
                    Some(rhs) => rhs,
                    None => panic!(
                        "Always needed, but set as optional when backward step is not required. {}",
                        "This avoids an uncessary clone, but if this panic, there is a bug!"
                    ),
                };

                let grad = grads.consume::<B, D>(&node_output);
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

                        grads.register::<B, D>(node, grad)
                    });

                node_rhs
                    .requirements([grad_4rhs, rhs_4rhs, lhs])
                    .run(|node, [grad, rhs, lhs]| {
                        let value = B::div(B::neg(lhs), B::powf(rhs, 2.0));
                        let grad = B::mul(grad, value);

                        grads.register::<B, D>(node, grad)
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
                node_output: NodeRef,
                grads: &mut Gradients,
                (shape, device, rhs): Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.run(|node, _| {
                    let ones = B::ones(shape, &device);
                    let tmp = B::div_scalar(ones, rhs);
                    let grad = B::mul(grad, tmp);

                    grads.register::<B, D>(node, grad)
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
                node_output: NodeRef,
                grads: &mut Gradients,
                (lhs, rhs): Self::State,
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume::<B, D>(&node_output));

                node_lhs
                    .requirements([grad_4lhs, rhs])
                    .run(|node, [grad, rhs]| {
                        let rhs = B::transpose(rhs);
                        let grad_lhs = B::matmul(grad, rhs);
                        grads.register::<B, D>(node, grad_lhs)
                    });

                node_rhs
                    .requirements([grad_4rhs, lhs])
                    .run(|node, [grad, lhs]| {
                        let lhs = B::transpose(lhs);
                        let grad_rhs = B::matmul(lhs, grad);
                        grads.register::<B, D>(node, grad_rhs)
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
                node_output: NodeRef,
                grads: &mut Gradients,
                (): Self::State,
            ) {
                let grad_out = grads.consume::<B, D>(&node_output);
                node.run(|node, _| grads.register::<B, D>(node, B::neg(grad_out)));
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
                node_output: NodeRef,
                grads: &mut Gradients,
                (dim1, dim2): Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.run(|node, _| {
                    let grad = B::swap_dims(grad, dim2, dim1);
                    grads.register::<B, D>(node, grad)
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
                node_output: NodeRef,
                grads: &mut Gradients,
                (shape_original, shape): Self::State,
            ) {
                let grad = grads.consume::<B, D2>(&node_output);

                node.run(|node, _| {
                    let shape_grad = B::shape(&grad);
                    let mut grad = grad;

                    for i in 0..D2 {
                        if shape.dims[i] == 1 && shape_grad.dims[i] != 1 {
                            grad = B::sum_dim(grad, i);
                        }
                    }

                    let grad = B::reshape(grad, shape_original);
                    grads.register::<B, D1>(node, grad)
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
                node_output: NodeRef,
                grads: &mut Gradients,
                (indexes, shape, device): Self::State,
            ) {
                let grad = grads.consume::<B, D1>(&node_output);

                node.run(|node, _| {
                    let zeros = B::zeros(shape, &device);
                    let grad = B::index_assign(zeros, indexes, grad);
                    grads.register::<B, D1>(node, grad)
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
        tensor: ADTensor<B, D1>,
        indexes: [std::ops::Range<usize>; D2],
        value: ADTensor<B, D1>,
    ) -> ADTensor<B, D1> {
        #[derive(Debug)]
        struct IndexAssign<const D2: usize>;

        impl<B: Backend, const D1: usize, const D2: usize> Backward<B, D1, 2> for IndexAssign<D2> {
            type State = ([std::ops::Range<usize>; D2], Shape<D1>, B::Device);

            fn backward(
                self,
                [node_lhs, node_rhs]: OpsNodes<2>,
                node_output: NodeRef,
                grads: &mut Gradients,
                (indexes, shape_rhs, device): Self::State,
            ) {
                let [grad_4lhs, grad_4rhs] =
                    duplicate([&node_lhs, &node_rhs], grads.consume::<B, D1>(&node_output));

                node_lhs.requirements([grad_4lhs]).run(|node, [grad]| {
                    let zeros = B::zeros(shape_rhs, &device);
                    let grad_lhs = B::index_assign(grad, indexes.clone(), zeros);
                    grads.register::<B, D1>(node, grad_lhs)
                });

                node_rhs.requirements([grad_4rhs]).run(|node, [grad]| {
                    let grad_rhs = B::index(grad, indexes);
                    grads.register::<B, D1>(node, grad_rhs)
                });
            }
        }

        IndexAssign.run(
            (
                indexes.clone(),
                B::shape(&value.primitive),
                B::device(&value.primitive),
            ),
            B::index_assign(tensor.primitive, indexes, value.primitive),
            [tensor.node, value.node],
            [tensor.graph, value.graph],
        )
    }

    fn mask_fill<const D: usize>(
        tensor: ADTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: Elem<B>,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MaskFill;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for MaskFill {
            type State = Option<BoolTensor<B, D>>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                mask: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([mask]).run(|node, [mask]| {
                    let grad = B::mask_fill(grad, mask, 0.to_elem());
                    grads.register::<B, D>(node, grad)
                });
            }
        }

        MaskFill.run(
            tensor.is_tracked().then(|| mask.clone()),
            B::mask_fill(tensor.primitive, mask, value),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::equal(lhs.primitive, rhs.primitive)
    }

    fn equal_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        B::equal_scalar(lhs.primitive, rhs)
    }

    fn greater<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::greater(lhs.primitive, rhs.primitive)
    }

    fn greater_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        B::greater_scalar(lhs.primitive, rhs)
    }

    fn greater_equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::greater_equal(lhs.primitive, rhs.primitive)
    }

    fn greater_equal_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        B::greater_equal_scalar(lhs.primitive, rhs)
    }

    fn lower<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::lower(lhs.primitive, rhs.primitive)
    }

    fn lower_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        B::lower_scalar(lhs.primitive, rhs)
    }

    fn lower_equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        B::lower_equal(lhs.primitive, rhs.primitive)
    }

    fn lower_equal_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        B::lower_equal_scalar(lhs.primitive, rhs)
    }

    fn detach<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        ADTensor::new(tensor.primitive)
    }

    fn mean<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        #[derive(Debug)]
        struct Mean<const D: usize>;

        impl<B: Backend, const D: usize> Backward<B, 1, 1> for Mean<D> {
            type State = Shape<D>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                shape: Self::State,
            ) {
                let grad = grads.consume::<B, 1>(&node_output);

                node.run(|node, _| {
                    let val = 1_f64 / shape.num_elements() as f64;
                    let ones = B::ones(shape, &B::device(&grad));
                    let val = B::mul_scalar(ones, val.to_elem());

                    let grad: Tensor<B, 1> = Tensor::from_primitive(grad);
                    let val: Tensor<B, D> = Tensor::from_primitive(val);

                    let grad = val.mul(grad.unsqueeze()).into_primitive();
                    grads.register::<B, D>(node, grad)
                });
            }
        }

        Mean.run(
            B::shape(&tensor.primitive),
            B::mean(tensor.primitive),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn sum<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        #[derive(Debug)]
        struct Sum<const D: usize>;

        impl<B: Backend, const D: usize> Backward<B, 1, 1> for Sum<D> {
            type State = Shape<D>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                shape: Self::State,
            ) {
                let grad = grads.consume::<B, 1>(&node_output);

                node.run(|node, _| {
                    let val = B::ones(shape, &B::device(&grad));

                    let grad: Tensor<B, 1> = Tensor::from_primitive(grad);
                    let val: Tensor<B, D> = Tensor::from_primitive(val);

                    let grad = val.mul(grad.unsqueeze()).into_primitive();
                    grads.register::<B, D>(node, grad)
                });
            }
        }

        Sum.run(
            B::shape(&tensor.primitive),
            B::sum(tensor.primitive),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn mean_dim<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MeamDim;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for MeamDim {
            type State = (Shape<D>, usize);

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                (shape, dim): Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.run(|node, _| {
                    let val = 1_f64 / shape.dims[dim] as f64;
                    let ones = B::ones(shape, &B::device(&grad));
                    let val = B::mul_scalar(ones, B::Elem::from_elem(val));

                    let grad = B::sum_dim(grad, dim);
                    let grad = B::mul(val, grad);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        MeamDim.run(
            (B::shape(&tensor.primitive), dim),
            B::mean_dim(tensor.primitive, dim),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn sum_dim<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct SumDim;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for SumDim {
            type State = (Shape<D>, usize);

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                (shape, dim): Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.run(|node, _| {
                    let ones = B::ones(shape, &B::device(&grad));

                    let grad = B::sum_dim(grad, dim);
                    let grad = B::mul(ones, grad);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        SumDim.run(
            (B::shape(&tensor.primitive), dim),
            B::sum_dim(tensor.primitive, dim),
            [tensor.node],
            [tensor.graph],
        )
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

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                _: Self::State,
            ) {
                let grad = grads.consume::<B::FullPrecisionBackend, D>(&node_output);

                node.run(|node, _| {
                    let grad = B::from_full_precision(grad);
                    grads.register::<B, D>(node, grad)
                });
            }
        }

        let ops = ToFullPrecision::<B> {
            phantom: PhantomData::default(),
        };
        ops.run(
            (),
            B::to_full_precision(&tensor.primitive),
            [tensor.node.clone()],
            [tensor.graph.clone()],
        )
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

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                _: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.run(|node, _| {
                    let grad = B::to_full_precision(&grad);
                    grads.register::<B::FullPrecisionBackend, D>(node, grad)
                });
            }
        }

        let ops = FromFullPrecision::<B::FullPrecisionBackend> {
            phantom: PhantomData::default(),
        };

        ops.run(
            (),
            B::from_full_precision(tensor.primitive),
            [tensor.node.clone()],
            [tensor.graph],
        )
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

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                output: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);
                node.run(|node, _| {
                    let grad = B::mul(grad, output);
                    grads.register::<B, D>(node, grad)
                });
            }
        }

        let output = B::exp(tensor.primitive);
        Exp.run(output.clone(), output, [tensor.node], [tensor.graph])
    }

    fn log<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Log;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Log {
            type State = B::TensorPrimitive<D>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                input: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.run(|node, _| {
                    let ones = B::ones(B::shape(&input), &B::device(&input));
                    let value = B::div(ones, input);
                    let grad = B::mul(grad, value);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        Log.run(
            tensor.primitive.clone(),
            B::log(tensor.primitive),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn log1p<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Log1P;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Log1P {
            type State = Option<B::TensorPrimitive<D>>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                input: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([input]).run(|node, [input]| {
                    let ones = B::ones(B::shape(&input), &B::device(&input));
                    let value = B::div(ones, B::add_scalar(input, 1.to_elem()));
                    let grad = B::mul(grad, value);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        Log1P.run(
            tensor.is_tracked().then(|| tensor.primitive.clone()),
            B::log1p(tensor.primitive),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn powf<const D: usize>(tensor: ADTensor<B, D>, value: f32) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct PowF;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for PowF {
            type State = (Option<B::TensorPrimitive<D>>, f32);

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                (tensor, value): Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([tensor]).run(|node, [tensor]| {
                    let tmp = B::powf(tensor, value - 1.0);
                    let value = B::mul_scalar(tmp, value.to_elem());
                    let grad = B::mul(grad, value);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        PowF.run(
            (tensor.is_tracked().then(|| tensor.primitive.clone()), value),
            B::powf(tensor.primitive, value),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn sqrt<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Sqrt;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Sqrt {
            type State = Option<B::TensorPrimitive<D>>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                input: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([input]).run(|node, [input]| {
                    let value = B::div_scalar(B::powf(input, -0.5), 2.to_elem());
                    let grad = B::mul(grad, value);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        Sqrt.run(
            tensor.is_tracked().then(|| tensor.primitive.clone()),
            B::sqrt(tensor.primitive),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn cos<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Cos;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Cos {
            type State = Option<B::TensorPrimitive<D>>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                input: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([input]).run(|node, [input]| {
                    let value = B::neg(B::sin(input));
                    let grad = B::mul(grad, value);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        Cos.run(
            tensor.is_tracked().then(|| tensor.primitive.clone()),
            B::cos(tensor.primitive),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn sin<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Sin;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Sin {
            type State = Option<B::TensorPrimitive<D>>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                input: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([input]).run(|node, [input]| {
                    let value = B::cos(input);
                    let grad = B::mul(grad, value);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        Sin.run(
            tensor.is_tracked().then(|| tensor.primitive.clone()),
            B::sin(tensor.primitive),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn tanh<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Tanh;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Tanh {
            type State = Option<B::TensorPrimitive<D>>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                output: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([output]).run(|node, [output]| {
                    let value = B::add_scalar(B::neg(B::powf(output, 2.0)), 1.to_elem());
                    let grad = B::mul(grad, value);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        let is_tracked = tensor.is_tracked();
        let output = B::tanh(tensor.primitive);

        Tanh.run(
            is_tracked.then(|| output.clone()),
            output,
            [tensor.node],
            [tensor.graph],
        )
    }

    fn erf<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Erf;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Erf {
            type State = Option<B::TensorPrimitive<D>>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                input: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([input]).run(|node, [input]| {
                    let exponent = B::neg(B::powf(input, 2.0));
                    let numerator = B::mul_scalar(B::exp(exponent), 2.0.to_elem());
                    let denominator = std::f64::consts::PI.sqrt().to_elem();
                    let value = B::div_scalar(numerator, denominator);
                    let grad = B::mul(grad, value);

                    grads.register::<B, D>(node, grad)
                });
            }
        }

        Erf.run(
            tensor.is_tracked().then(|| tensor.primitive.clone()),
            B::erf(tensor.primitive),
            [tensor.node],
            [tensor.graph],
        )
    }

    fn cat<const D: usize>(tensors: Vec<ADTensor<B, D>>, dim: usize) -> ADTensor<B, D> {
        #[derive(new, Debug)]
        struct CatStep<B: Backend, const D: usize> {
            nodes: Vec<OpsNode<(), 0>>,
            output: NodeRef,
            phantom: PhantomData<B>,
            dim: usize,
        }

        impl<B: Backend, const D: usize> Step for CatStep<B, D> {
            fn step(self: Box<Self>, grads: &mut Gradients) {
                let grad = grads.consume::<B, D>(&self.output);
                let indexes: Vec<_> = B::shape(&grad).dims.iter().map(|v| 0..*v).collect();
                let indexes: [std::ops::Range<usize>; D] = indexes.try_into().unwrap();

                self.nodes.into_iter().enumerate().for_each(|(i, node)| {
                    node.run(|node, _| {
                        let mut indexes = indexes.clone();
                        indexes[self.dim] = i..i + 1;
                        grads.register::<B, D>(node, B::index(grad.clone(), indexes));
                    });
                });
            }

            fn node(&self) -> NodeRef {
                self.output.clone()
            }
        }

        let is_tracked = tensors
            .iter()
            .map(|tensor| tensor.is_tracked())
            .reduce(|acc, is_tracked| is_tracked || acc)
            .unwrap_or(false);

        let mut nodes = Vec::with_capacity(tensors.len());
        let mut graphs = Vec::with_capacity(tensors.len());
        let mut primitives = Vec::with_capacity(tensors.len());

        tensors.into_iter().for_each(|tensor| {
            nodes.push(tensor.node);
            primitives.push(tensor.primitive);
            graphs.push(tensor.graph);
        });

        let requirement = Requirement::from_nodes(&nodes);
        let output = B::cat(primitives, dim);
        let output = ADTensor::from_ops(&nodes, output, graphs.into_iter(), requirement);

        let nodes = nodes
            .into_iter()
            .map(|node| match node.clone_if_require_grad() {
                Some(node) => OpsNode::Tracked(node, []),
                None => OpsNode::Untrack,
            })
            .collect::<Vec<_>>();

        if !is_tracked {
            return output;
        }

        let ops = CatStep::<B, D>::new(nodes, output.node.clone(), dim);
        output.register_ops(ops)
    }

    fn relu<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Relu;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Relu {
            type State = Option<B::TensorPrimitive<D>>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                node_output: NodeRef,
                grads: &mut Gradients,
                output: Self::State,
            ) {
                let grad = grads.consume::<B, D>(&node_output);

                node.requirements([output]).run(|node, [output]| {
                    let zero = 0.to_elem();
                    let mask = B::lower_equal_scalar(output, zero);
                    let grad = B::mask_fill(grad, mask, zero);

                    grads.register::<B, D>(node, grad)
                });
            }
        }
        let is_tracked = tensor.is_tracked();
        let output = B::relu(tensor.primitive);

        Relu.run(
            is_tracked.then(|| output.clone()),
            output,
            [tensor.node],
            [tensor.graph],
        )
    }
}
