use super::{binary_ops_wrapper, unary_ops_wrapper};
use crate::{
    backend::{
        autodiff::{ADBackendDecorator, ADTensor},
        Backend,
    },
    graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState},
    ops::{Ones, TensorOps, TensorOpsAggregation},
    Data, Shape,
};

impl<B: Backend, const D: usize> std::ops::Add<ADTensor<D, B>> for ADTensor<D, B> {
    type Output = ADTensor<D, B>;

    fn add(self, rhs: Self) -> Self::Output {
        ADBackendDecorator::add(&self, &rhs)
    }
}

impl<B: Backend> TensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn shape<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> &Shape<D> {
        B::shape(tensor.tensor_ref())
    }

    fn to_data<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> Data<<ADBackendDecorator<B> as Backend>::Elem, D> {
        B::to_data(tensor.tensor_ref())
    }

    fn into_data<const D: usize>(
        tensor: <ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> Data<<ADBackendDecorator<B> as Backend>::Elem, D> {
        B::into_data(tensor.tensor())
    }

    fn bool_shape<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> &Shape<D> {
        B::bool_shape(tensor)
    }

    fn bool_to_data<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        B::bool_to_data(tensor)
    }

    fn bool_into_data<const D: usize>(
        tensor: <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        B::bool_into_data(tensor)
    }

    fn device<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::Device {
        B::device(tensor.tensor_ref())
    }

    fn to_device<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        device: <ADBackendDecorator<B> as Backend>::Device,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        struct ToDeviceBackward<B: Backend, const D: usize> {
            device: B::Device,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for ToDeviceBackward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                B::to_device(&state.output.grad(), self.device)
            }
        }

        let device_old = B::device(tensor.tensor_ref());
        let input = tensor.node.clone();
        let output = B::to_device(tensor.tensor_ref(), device);
        let ops = ToDeviceBackward::<B, D>::new(device_old);

        unary_ops_wrapper(input, output, ops)
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: <ADBackendDecorator<B> as Backend>::Device,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        ADTensor::from_tensor(B::empty(shape, device))
    }

    fn add<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct AddBackward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize>
            BinaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for AddBackward<B, D>
        {
            fn partial_left(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                state.output.grad()
            }

            fn partial_right(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                state.output.grad()
            }
        }

        let output = B::add(lhs.tensor_ref(), rhs.tensor_ref());
        let ops = AddBackward::<B, D>::default();

        binary_ops_wrapper(lhs.node.clone(), rhs.node.clone(), output, ops)
    }

    fn add_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct AddScalarBackward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for AddScalarBackward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                state.output.grad()
            }
        }

        let output = B::add_scalar(lhs.tensor_ref(), rhs);
        let ops = AddScalarBackward::<B, D>::default();

        unary_ops_wrapper(lhs.node.clone(), output, ops)
    }

    fn sub<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct SubBackward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize>
            BinaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for SubBackward<B, D>
        {
            fn partial_left(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                state.output.grad()
            }

            fn partial_right(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                B::neg(&state.output.grad())
            }
        }

        let output = B::sub(lhs.tensor_ref(), rhs.tensor_ref());
        let ops = SubBackward::<B, D>::default();

        binary_ops_wrapper(lhs.node.clone(), rhs.node.clone(), output, ops)
    }

    fn sub_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct SubScalarBackward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for SubScalarBackward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                state.output.grad()
            }
        }

        let output = B::sub_scalar(lhs.tensor_ref(), rhs);
        let ops = SubScalarBackward::<B, D>::default();

        unary_ops_wrapper(lhs.node.clone(), output, ops)
    }

    fn mul<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct MulBackward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize>
            BinaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for MulBackward<B, D>
        {
            fn partial_left(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                B::mul(&state.output.grad(), &state.right.value())
            }

            fn partial_right(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                B::mul(&state.output.grad(), &state.left.value())
            }
        }

        let output = B::mul(lhs.tensor_ref(), rhs.tensor_ref());
        let ops = MulBackward::<B, D>::default();

        binary_ops_wrapper(lhs.node.clone(), rhs.node.clone(), output, ops)
    }

    fn mul_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        struct MulScalarBackward<B: Backend, const D: usize> {
            elem: B::Elem,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for MulScalarBackward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                B::mul_scalar(&state.output.grad(), &self.elem)
            }
        }

        let output = B::mul_scalar(lhs.tensor_ref(), rhs);
        let ops = MulScalarBackward::<B, D>::new(*rhs);

        unary_ops_wrapper(lhs.node.clone(), output, ops)
    }

    fn div<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct DivBackward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize>
            BinaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for DivBackward<B, D>
        {
            fn partial_left(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                let value = state.right.value();
                let value = B::div(&value.ones(), &value);

                B::mul(&state.output.grad(), &value)
            }

            fn partial_right(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                let value_left = state.left.value();
                let value_right = state.right.value();
                let value = B::div(&B::neg(&value_left), &B::mul(&value_right, &value_right));

                B::mul(&state.output.grad(), &value)
            }
        }

        let output = B::div(lhs.tensor_ref(), rhs.tensor_ref());
        let ops = DivBackward::<B, D>::default();

        binary_ops_wrapper(lhs.node.clone(), rhs.node.clone(), output, ops)
    }

    fn div_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        struct DivScalarBackward<B: Backend, const D: usize> {
            elem: B::Elem,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for DivScalarBackward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                let value = state.input.value();
                let tmp = B::div_scalar(&value.ones(), &self.elem);

                B::mul(&state.output.grad(), &tmp)
            }
        }

        let output = B::div_scalar(lhs.tensor_ref(), rhs);
        let ops = DivScalarBackward::<B, D>::new(*rhs);

        unary_ops_wrapper(lhs.node.clone(), output, ops)
    }

    fn matmul<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct MatmulBackward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize>
            BinaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for MatmulBackward<B, D>
        {
            fn partial_left(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                let out_grad = state.output.grad();
                let rhs = B::transpose(&state.right.value());
                B::matmul(&out_grad, &rhs)
            }

            fn partial_right(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                let out_grad = state.output.grad();
                let lhs = B::transpose(&state.left.value());
                B::matmul(&lhs, &out_grad)
            }
        }

        let output = B::matmul(lhs.tensor_ref(), rhs.tensor_ref());
        let ops = MatmulBackward::<B, D>::default();

        binary_ops_wrapper(lhs.node.clone(), rhs.node.clone(), output, ops)
    }

    fn neg<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct NegBackward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for NegBackward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                B::neg(&state.output.grad())
            }
        }

        let output = B::neg(tensor.tensor_ref());
        let ops = NegBackward::<B, D>::default();

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn swap_dims<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        struct SwapDimsBackward<B: Backend, const D: usize> {
            _b: B,
            dim1: usize,
            dim2: usize,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for SwapDimsBackward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                B::swap_dims(&state.output.grad(), self.dim2, self.dim1)
            }
        }

        let output = B::swap_dims(tensor.tensor_ref(), dim1, dim2);
        let ops = SwapDimsBackward::<B, D>::new(B::default(), dim1, dim2);

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D2> {
        #[derive(new, Debug)]
        struct ReshapeBackward<B: Backend, const D1: usize, const D2: usize> {
            shape: Shape<D1>,
            _b: B,
        }

        impl<B: Backend, const D1: usize, const D2: usize>
            UnaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D2>>
            for ReshapeBackward<B, D1, D2>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D1>, B::TensorPrimitive<D2>>,
            ) -> B::TensorPrimitive<D1> {
                let mut grad = state.output.grad();
                let value = state.output.value();

                let shape_grad = *B::shape(&grad);
                let shape_value = *B::shape(&value);

                if shape_value == shape_grad {
                    return B::reshape(&grad, self.shape);
                }

                for i in 0..D2 {
                    if shape_value.dims[i] == 1 && shape_grad.dims[i] != 1 {
                        grad = grad.sum_dim(i);
                    }
                }

                B::reshape(&grad, self.shape)
            }
        }

        let shape_old = B::shape(tensor.tensor_ref());
        let output = B::reshape(tensor.tensor_ref(), shape);
        let ops = ReshapeBackward::<B, D1, D2>::new(*shape_old, B::default());

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }
}
