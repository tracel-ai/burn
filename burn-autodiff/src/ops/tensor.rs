use crate::{
    grads::Gradients,
    graph::{NodeRef, Requirement},
    ops::Ops,
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

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        ADTensor::new(B::empty(shape, device))
    }

    fn add<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Add;

        impl<B: Backend, const D: usize> Ops<B, D, 2> for Add {
            type Forward = (B::TensorPrimitive<D>, B::TensorPrimitive<D>);
            type Backward = ();

            fn forward(&self, (lhs, rhs): Self::Forward) -> <B as Backend>::TensorPrimitive<D> {
                B::add(lhs, rhs)
            }
            fn backward(
                self,
                [node_lhs, node_rhs]: [Option<NodeRef>; 2],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _: Self::Backward,
            ) {
                let grad_out = grads.consume(&output);
                let [grad_out_lhs, grad_out_rhs] =
                    duplicate([node_lhs.is_some(), node_rhs.is_some()], grad_out);

                if let Some((node_lhs, grad)) = node_lhs.zip(grad_out_lhs) {
                    grads.update(node_lhs, grad)
                }
                if let Some((node_rhs, grad)) = node_rhs.zip(grad_out_rhs) {
                    grads.update(node_rhs, grad)
                }
            }
        }

        Add.run(
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
            (lhs.primitive, rhs.primitive),
            (),
        )
    }

    fn add_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct AddScalar;

        impl<B: Backend, const D: usize> Ops<B, D, 1> for AddScalar {
            type Forward = (B::TensorPrimitive<D>, Elem<B>);
            type Backward = ();

            fn forward(&self, (lhs, rhs): Self::Forward) -> B::TensorPrimitive<D> {
                B::add_scalar(lhs, rhs)
            }
            fn backward(
                self,
                [node]: [Option<NodeRef>; 1],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _: (),
            ) {
                let grad_out = grads.consume(&output);

                if let Some(node) = node {
                    grads.update(node, grad_out)
                }
            }
        }

        AddScalar.run([lhs.node], [lhs.graph], (lhs.primitive, rhs), ())
    }

    fn sub<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Sub;

        impl<B: Backend, const D: usize> Ops<B, D, 2> for Sub {
            type Forward = (B::TensorPrimitive<D>, B::TensorPrimitive<D>);
            type Backward = ();

            fn forward(&self, (lhs, rhs): Self::Forward) -> B::TensorPrimitive<D> {
                B::sub(lhs, rhs)
            }
            fn backward(
                self,
                [node_lhs, node_rhs]: [Option<NodeRef>; 2],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _: (),
            ) {
                let grad_out = grads.consume(&output);
                let [grad_out_lhs, grad_out_rhs] =
                    duplicate([node_lhs.is_some(), node_rhs.is_some()], grad_out);

                if let Some((node_lhs, grad)) = node_lhs.zip(grad_out_lhs) {
                    grads.update(node_lhs, grad)
                }
                if let Some((node_rhs, grad)) = node_rhs.zip(grad_out_rhs) {
                    grads.update(node_rhs, B::neg(grad))
                }
            }
        }

        Sub.run(
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
            (lhs.primitive, rhs.primitive),
            (),
        )
    }

    fn sub_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct SubScalar;

        impl<B: Backend, const D: usize> Ops<B, D, 1> for SubScalar {
            type Forward = (B::TensorPrimitive<D>, Elem<B>);
            type Backward = ();

            fn forward(&self, (lhs, rhs): Self::Forward) -> B::TensorPrimitive<D> {
                B::sub_scalar(lhs, rhs)
            }
            fn backward(
                self,
                [node]: [Option<NodeRef>; 1],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _: (),
            ) {
                let grad_out = grads.consume(&output);

                if let Some(node) = node {
                    grads.update(node, grad_out)
                }
            }
        }

        SubScalar.run([lhs.node], [lhs.graph], (lhs.primitive, rhs), ())
    }

    fn mul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Mul;

        impl<B: Backend, const D: usize> Ops<B, D, 2> for Mul {
            type Forward = (B::TensorPrimitive<D>, B::TensorPrimitive<D>);
            type Backward = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn forward(&self, (lhs, rhs): Self::Forward) -> B::TensorPrimitive<D> {
                B::mul(lhs, rhs)
            }
            fn backward(
                self,
                [node_lhs, node_rhs]: [Option<NodeRef>; 2],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (lhs, rhs): Self::Backward,
            ) {
                let grad_out = grads.consume(&output);
                let [grad_out_lhs, grad_out_rhs] =
                    duplicate([node_lhs.is_some(), node_rhs.is_some()], grad_out);

                if let Some((node_lhs, grad_out)) = node_lhs.zip(grad_out_lhs) {
                    let grad_lhs = B::mul(grad_out, rhs.unwrap());
                    grads.update(node_lhs, grad_lhs)
                }

                if let Some((node_rhs, grad_out)) = node_rhs.zip(grad_out_rhs) {
                    let grad_rhs = B::mul(grad_out, lhs.unwrap());
                    grads.update(node_rhs, grad_rhs)
                }
            }
        }

        let state_lhs = match rhs.node.requirement {
            Requirement::None => None,
            _ => Some(lhs.primitive.clone()),
        };
        let state_rhs = match lhs.node.requirement {
            Requirement::None => None,
            _ => Some(rhs.primitive.clone()),
        };
        Mul.run(
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
            (lhs.primitive, rhs.primitive),
            (state_lhs, state_rhs),
        )
    }

    fn mul_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MulScalar;

        impl<B: Backend, const D: usize> Ops<B, D, 1> for MulScalar {
            type Forward = (B::TensorPrimitive<D>, Elem<B>);
            type Backward = Elem<B>;

            fn forward(&self, (lhs, rhs): Self::Forward) -> B::TensorPrimitive<D> {
                B::mul_scalar(lhs, rhs)
            }
            fn backward(
                self,
                [node]: [Option<NodeRef>; 1],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                rhs: Elem<B>,
            ) {
                let grad_out = grads.consume(&output);

                if let Some(node) = node {
                    let grad = B::mul_scalar(grad_out, rhs);
                    grads.update(node, grad)
                }
            }
        }

        MulScalar.run([lhs.node], [lhs.graph], (lhs.primitive, rhs), rhs)
    }

    fn div<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Div;

        impl<B: Backend, const D: usize> Ops<B, D, 2> for Div {
            type Forward = (B::TensorPrimitive<D>, B::TensorPrimitive<D>);
            type Backward = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn forward(&self, (lhs, rhs): Self::Forward) -> B::TensorPrimitive<D> {
                B::mul(lhs, rhs)
            }
            fn backward(
                self,
                [node_lhs, node_rhs]: [Option<NodeRef>; 2],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (lhs, rhs): Self::Backward,
            ) {
                let grad_out = grads.consume(&output);

                let [grad_out_lhs, grad_out_rhs] =
                    duplicate([node_lhs.is_some(), node_rhs.is_some()], grad_out);
                let [rhs_lhs, rhs_rhs] =
                    duplicate([node_lhs.is_some(), node_rhs.is_some()], rhs.unwrap());

                if let Some((node_lhs, grad_out)) = node_lhs.zip(grad_out_lhs) {
                    let rhs = rhs_lhs.unwrap();
                    let device = B::device(&rhs);
                    let shape = B::shape(&rhs);
                    let ones = B::ones(shape, &device);
                    let value = B::div(ones, rhs);
                    let grad = B::mul(grad_out, value);

                    grads.update(node_lhs, grad)
                }

                if let Some((node_rhs, grad_out)) = node_rhs.zip(grad_out_rhs) {
                    let rhs = rhs_rhs.unwrap();
                    let value = B::div(B::neg(lhs.unwrap()), B::powf(rhs, 2.0));
                    let grad = B::mul(grad_out, value);

                    grads.update(node_rhs, grad)
                }
            }
        }

        let state_lhs = match rhs.node.requirement {
            Requirement::None => None,
            _ => Some(lhs.primitive.clone()),
        };
        let state_rhs = match !lhs.node.requirement.is_none() || !lhs.node.requirement.is_none() {
            true => Some(rhs.primitive.clone()),
            _ => None,
        };

        Div.run(
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
            (lhs.primitive, rhs.primitive),
            (state_lhs, state_rhs),
        )
    }

    fn div_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct DivScalar;

        impl<B: Backend, const D: usize> Ops<B, D, 1> for DivScalar {
            type Forward = (B::TensorPrimitive<D>, Elem<B>);
            type Backward = (Shape<D>, B::Device, Elem<B>);

            fn forward(&self, (lhs, rhs): Self::Forward) -> B::TensorPrimitive<D> {
                B::div_scalar(lhs, rhs)
            }
            fn backward(
                self,
                [node]: [Option<NodeRef>; 1],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (shape, device, rhs): Self::Backward,
            ) {
                let grad_out = grads.consume(&output);

                if let Some(node) = node {
                    let ones = B::ones(shape, &device);
                    let tmp = B::div_scalar(ones, rhs);
                    let grad = B::mul(grad_out, tmp);

                    grads.update(node, grad)
                }
            }
        }

        let device = B::device(&lhs.primitive);
        let shape = B::shape(&lhs.primitive);

        DivScalar.run(
            [lhs.node],
            [lhs.graph],
            (lhs.primitive, rhs),
            (shape, device, rhs),
        )
    }

    fn matmul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Matmul;

        impl<B: Backend, const D: usize> Ops<B, D, 2> for Matmul {
            type Forward = (B::TensorPrimitive<D>, B::TensorPrimitive<D>);
            type Backward = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn forward(&self, (lhs, rhs): Self::Forward) -> B::TensorPrimitive<D> {
                B::matmul(lhs, rhs)
            }
            fn backward(
                self,
                [node_lhs, node_rhs]: [Option<NodeRef>; 2],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (lhs, rhs): Self::Backward,
            ) {
                let grad_out = grads.consume(&output);

                let [grad_out_lhs, grad_out_rhs] =
                    duplicate([node_lhs.is_some(), node_rhs.is_some()], grad_out);

                if let Some((node_lhs, grad_out)) = node_lhs.zip(grad_out_lhs) {
                    let rhs = B::transpose(rhs.unwrap());
                    let grad_lhs = B::matmul(grad_out, rhs);
                    grads.update(node_lhs, grad_lhs)
                }
                if let Some((node_rhs, grad_out)) = node_rhs.zip(grad_out_rhs) {
                    let lhs = B::transpose(lhs.unwrap());
                    let grad_rhs = B::matmul(lhs, grad_out);
                    grads.update(node_rhs, grad_rhs)
                }
            }
        }

        let state_lhs = match rhs.node.requirement {
            Requirement::None => None,
            _ => Some(lhs.primitive.clone()),
        };
        let state_rhs = match lhs.node.requirement {
            Requirement::None => None,
            _ => Some(rhs.primitive.clone()),
        };

        Matmul.run(
            [lhs.node, rhs.node],
            [lhs.graph, rhs.graph],
            (lhs.primitive, rhs.primitive),
            (state_lhs, state_rhs),
        )
    }

    fn neg<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Neg;

        impl<B: Backend, const D: usize> Ops<B, D, 1> for Neg {
            type Forward = B::TensorPrimitive<D>;
            type Backward = ();

            fn forward(&self, tensor: Self::Forward) -> B::TensorPrimitive<D> {
                B::neg(tensor)
            }
            fn backward(
                self,
                [node]: [Option<NodeRef>; 1],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (): Self::Backward,
            ) {
                let grad_out = grads.consume(&output);

                if let Some(node) = node {
                    let grad = B::neg(grad_out);
                    grads.update(node, grad)
                }
            }
        }

        Neg.run([tensor.node], [tensor.graph], tensor.primitive, ())
    }

    fn swap_dims<const D: usize>(
        tensor: ADTensor<B, D>,
        dim1: usize,
        dim2: usize,
    ) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct SwapDim;

        impl<B: Backend, const D: usize> Ops<B, D, 1> for SwapDim {
            type Forward = (B::TensorPrimitive<D>, usize, usize);
            type Backward = (usize, usize);

            fn forward(&self, (tensor, dim1, dim2): Self::Forward) -> B::TensorPrimitive<D> {
                B::swap_dims(tensor, dim1, dim2)
            }
            fn backward(
                self,
                [node]: [Option<NodeRef>; 1],
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (dim1, dim2): Self::Backward,
            ) {
                let grad_out = grads.consume(&output);

                if let Some(node) = node {
                    let grad = B::swap_dims(grad_out, dim2, dim1);
                    grads.update(node, grad)
                }
            }
        }

        SwapDim.run(
            [tensor.node],
            [tensor.graph],
            (tensor.primitive, dim1, dim2),
            (dim1, dim2),
        )
    }

    fn reshape<const D1: usize, const D2: usize>(
        _tensor: ADTensor<B, D1>,
        _shape: Shape<D2>,
    ) -> ADTensor<B, D2> {
        todo!()
    }

    fn index<const D1: usize, const D2: usize>(
        _tensor: ADTensor<B, D1>,
        _indexes: [std::ops::Range<usize>; D2],
    ) -> ADTensor<B, D1> {
        todo!()
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

    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        Self::from_data(Data::zeros(shape), device)
    }

    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        Self::from_data(Data::ones(shape), device)
    }

    fn arange(range: std::ops::Range<usize>, device: &B::Device) -> IntTensor<B, 1> {
        B::arange(range, device)
    }
}
