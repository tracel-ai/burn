use super::{binary_ops_wrapper, unary_ops_wrapper};
use crate::graph::converter::Forward2BackwardGraphConverter;
use crate::graph::node::{BackwardNode, BackwardNodeRef, BackwardNodeState, ForwardNodeRef};
use crate::graph::ops::*;
use crate::ops::unary_ops_wrapper_explicit;
use crate::tensor::ADTensor;
use crate::ADBackendDecorator;
use burn_tensor::backend::Backend;
use burn_tensor::{ops::*, Data, Distribution, ElementConversion, Shape, Tensor};
use std::ops::Range;
use std::sync::Arc;

impl<B: Backend, const D: usize> std::ops::Add<ADTensor<D, B>> for ADTensor<D, B> {
    type Output = ADTensor<D, B>;

    fn add(self, rhs: Self) -> Self::Output {
        ADBackendDecorator::add(&self, &rhs)
    }
}

impl<B: Backend> TensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn from_data<const D: usize>(data: Data<B::Elem, D>, device: B::Device) -> ADTensor<D, B> {
        let tensor = B::from_data(data, device);
        ADTensor::from_tensor(tensor)
    }

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        device: B::Device,
    ) -> B::BoolTensorPrimitive<D> {
        B::from_data_bool(data, device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<B::Elem>,
        device: B::Device,
    ) -> ADTensor<D, B> {
        ADTensor::from_tensor(B::random(shape, distribution, device))
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: B::Device) -> ADTensor<D, B> {
        ADTensor::from_tensor(B::zeros(shape, device))
    }

    fn ones<const D: usize>(shape: Shape<D>, device: B::Device) -> ADTensor<D, B> {
        ADTensor::from_tensor(B::ones(shape, device))
    }

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

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D2> {
        B::bool_reshape(tensor, shape)
    }

    fn bool_index<const D1: usize, const D2: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D1> {
        B::bool_index(tensor, indexes)
    }

    fn bool_to_device<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>,
        device: <ADBackendDecorator<B> as Backend>::Device,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::bool_to_device(tensor, device)
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
                        grad = B::sum_dim(&grad, i);
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

    fn index<const D1: usize, const D2: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D1> {
        #[derive(new, Debug)]
        struct IndexBackward<B: Backend, const D1: usize, const D2: usize> {
            indexes: [Range<usize>; D2],
            _b: B,
        }

        impl<B: Backend, const D1: usize, const D2: usize>
            UnaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D1>> for IndexBackward<B, D1, D2>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D1>, B::TensorPrimitive<D1>>,
            ) -> B::TensorPrimitive<D1> {
                B::index_assign(
                    &state.input.value().zeros(),
                    self.indexes.clone(),
                    &state.output.grad(),
                )
            }
        }

        let output = B::index(tensor.tensor_ref(), indexes.clone());
        let ops = IndexBackward::<B, D1, D2>::new(indexes, B::default());

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D1>,
        indexes: [Range<usize>; D2],
        value: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D1>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D1> {
        #[derive(new, Debug)]
        struct IndexAssignBackward<B: Backend, const D1: usize, const D2: usize> {
            indexes: [Range<usize>; D2],
            _b: B,
        }

        impl<B: Backend, const D1: usize, const D2: usize>
            BinaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D1>, B::TensorPrimitive<D1>>
            for IndexAssignBackward<B, D1, D2>
        {
            fn partial_left(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D1>,
                    B::TensorPrimitive<D1>,
                    B::TensorPrimitive<D1>,
                >,
            ) -> B::TensorPrimitive<D1> {
                B::index_assign(
                    &state.output.grad(),
                    self.indexes.clone(),
                    &state.right.value().zeros(),
                )
            }

            fn partial_right(
                &self,
                state: &BinaryOpsNodeState<
                    B::TensorPrimitive<D1>,
                    B::TensorPrimitive<D1>,
                    B::TensorPrimitive<D1>,
                >,
            ) -> B::TensorPrimitive<D1> {
                B::index(&state.output.grad(), self.indexes.clone())
            }
        }

        let output = B::index_assign(tensor.tensor_ref(), indexes.clone(), value.tensor_ref());
        let ops = IndexAssignBackward::<B, D1, D2>::new(indexes, B::default());

        binary_ops_wrapper(tensor.node.clone(), value.node.clone(), output, ops)
    }

    fn mask_fill<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        mask: &<ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>,
        value: <ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        struct MaskFillBackward<B: Backend, const D: usize> {
            mask: B::BoolTensorPrimitive<D>,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for MaskFillBackward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                B::mask_fill(
                    &state.output.grad(),
                    &self.mask,
                    B::Elem::zeros(&B::Elem::default()),
                )
            }
        }

        let output = B::mask_fill(tensor.tensor_ref(), mask, value);
        let ops = MaskFillBackward::<B, D>::new(mask.clone());

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn equal<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::equal(lhs.tensor_ref(), rhs.tensor_ref())
    }

    fn equal_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::equal_scalar(lhs.tensor_ref(), rhs)
    }

    fn greater<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::greater(lhs.tensor_ref(), rhs.tensor_ref())
    }

    fn greater_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::greater_scalar(lhs.tensor_ref(), rhs)
    }

    fn greater_equal<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::greater_equal(lhs.tensor_ref(), rhs.tensor_ref())
    }

    fn greater_equal_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::greater_equal_scalar(lhs.tensor_ref(), rhs)
    }

    fn lower<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::lower(lhs.tensor_ref(), rhs.tensor_ref())
    }

    fn lower_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::lower_scalar(lhs.tensor_ref(), rhs)
    }

    fn lower_equal<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::lower_equal(lhs.tensor_ref(), rhs.tensor_ref())
    }

    fn lower_equal_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D> {
        B::lower_equal_scalar(lhs.tensor_ref(), rhs)
    }

    fn detach<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        ADTensor::from_tensor(B::detach(tensor.tensor_ref()))
    }

    fn mean<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<1> {
        #[derive(new, Debug)]
        struct Backward<B: Backend, const D: usize> {
            shape: Shape<D>,
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<1>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<1>>,
            ) -> B::TensorPrimitive<D> {
                let grad = state.output.grad();
                let ones = B::ones(self.shape, B::device(&grad));

                let grad: Tensor<B, 1> = Tensor::from_primitive(grad);
                let val = 1_f64 / self.shape.num_elements() as f64;
                let ones: Tensor<B, D> = Tensor::from_primitive(ones).mul_scalar(val);

                ones.mul(&grad.unsqueeze()).into_primitive()
            }
        }

        let shape = B::shape(tensor.tensor_ref());
        let output = B::mean(tensor.tensor_ref());
        let ops = Backward::<B, D>::new(*shape, B::default());

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn sum<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<1> {
        #[derive(new, Debug)]
        struct Backward<B: Backend, const D: usize> {
            shape: Shape<D>,
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<1>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<1>>,
            ) -> B::TensorPrimitive<D> {
                let grad = state.output.grad();
                let ones = B::ones(self.shape, B::device(&grad));

                let grad: Tensor<B, 1> = Tensor::from_primitive(grad);
                let ones: Tensor<B, D> = Tensor::from_primitive(ones);

                ones.mul(&grad.unsqueeze()).into_primitive()
            }
        }

        let shape = B::shape(tensor.tensor_ref());
        let output = B::sum(tensor.tensor_ref());
        let ops = Backward::<B, D>::new(*shape, B::default());

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn mean_dim<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        struct Backward<B: Backend, const D: usize> {
            shape: Shape<D>,
            dim: usize,
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                let grad = B::sum_dim(&state.output.grad(), self.dim);
                let ones = B::ones(self.shape, B::device(&grad));

                let val = 1_f64 / self.shape.dims[self.dim] as f64;
                let ones = B::mul_scalar(&ones, &B::Elem::from_elem(val));

                B::mul(&ones, &grad)
            }
        }

        let shape = B::shape(tensor.tensor_ref());
        let output = B::mean_dim(tensor.tensor_ref(), dim);
        let ops = Backward::<B, D>::new(*shape, dim, B::default());

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn sum_dim<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        struct Backward<B: Backend, const D: usize> {
            shape: Shape<D>,
            dim: usize,
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                let grad = B::sum_dim(&state.output.grad(), self.dim);
                let ones = B::ones(self.shape, B::device(&grad));

                B::mul(&ones, &grad)
            }
        }

        let shape = B::shape(tensor.tensor_ref());
        let output = B::sum_dim(tensor.tensor_ref(), dim);
        let ops = Backward::<B, D>::new(*shape, dim, B::default());

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn to_full_precision<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <<ADBackendDecorator<B> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>
    {
        #[derive(Default, Debug)]
        struct Backward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize>
            UnaryOps<
                B::TensorPrimitive<D>,
                <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
            > for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<
                    B::TensorPrimitive<D>,
                    <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
                >,
            ) -> B::TensorPrimitive<D> {
                let grad = state.output.grad();
                B::from_full_precision(&grad)
            }
        }

        let output = B::to_full_precision(tensor.tensor_ref());
        let ops = Backward::<B, D>::default();

        unary_ops_wrapper_explicit::<B, B::FullPrecisionBackend, Backward<B, D>, D, D>(
            tensor.node.clone(),
            output,
            ops,
        )
    }

    fn from_full_precision<const D: usize>(
        tensor: &<<ADBackendDecorator<B> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct Backward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize>
            UnaryOps<
                <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
                B::TensorPrimitive<D>,
            > for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<
                    <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
                    B::TensorPrimitive<D>,
                >,
            ) -> <B::FullPrecisionBackend as Backend>::TensorPrimitive<D> {
                let grad = state.output.grad();
                B::to_full_precision(&grad)
            }
        }

        let output = B::from_full_precision(tensor.tensor_ref());
        let ops = Backward::<B, D>::default();

        unary_ops_wrapper_explicit::<B::FullPrecisionBackend, B, Backward<B, D>, D, D>(
            tensor.node.clone(),
            output,
            ops,
        )
    }

    fn argmax<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <<ADBackendDecorator<B> as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
        B::argmax(tensor.tensor_ref(), dim)
    }

    fn argmin<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <<ADBackendDecorator<B> as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
        B::argmin(tensor.tensor_ref(), dim)
    }

    fn exp<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct Backward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                B::mul(&state.output.grad(), &state.output.value())
            }
        }

        let output = B::exp(tensor.tensor_ref());
        let ops = Backward::<B, D>::default();

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn log<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct Backward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                let value = state.input.value();
                let value = B::div(&value.ones(), &value);
                B::mul(&state.output.grad(), &value)
            }
        }

        let output = B::log(tensor.tensor_ref());
        let ops = Backward::<B, D>::default();

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn powf<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        value: f32,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        struct Backward<B: Backend, const D: usize> {
            value: f32,
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                let value = B::mul_scalar(
                    &B::powf(&state.input.value(), self.value - 1.0),
                    &self.value.clone().to_elem(),
                );
                B::mul(&state.output.grad(), &value)
            }
        }

        let output = B::powf(tensor.tensor_ref(), value);
        let ops = Backward::<B, D>::new(value, B::default());

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn erf<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct Backward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                let value = state.input.value();
                let exponent = B::neg(&B::powf(&value, 2.0));
                let numerator = B::mul_scalar(&B::exp(&exponent), &2.0.to_elem());
                let denominator = std::f64::consts::PI.sqrt().to_elem();
                let value = B::div_scalar(&numerator, &denominator);

                B::mul(&state.output.grad(), &value)
            }
        }

        let output = B::erf(tensor.tensor_ref());
        let ops = Backward::<B, D>::default();

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }

    fn cat<const D: usize>(
        tensors: &[<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>],
        dim: usize,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(new, Debug)]
        pub struct ForwardCatOps<const D: usize, B: Backend> {
            nodes: Vec<ForwardNodeRef<B::TensorPrimitive<D>>>,
            dim: usize,
        }

        #[derive(new, Debug)]
        pub struct BackwardCatOps<const D: usize, B: Backend> {
            nodes: Vec<BackwardNodeRef<B::TensorPrimitive<D>>>,
            dim: usize,
        }

        impl<const D: usize, B: Backend> ForwardRecordedOps<B::TensorPrimitive<D>> for ForwardCatOps<D, B> {
            fn to_backward(
                &self,
                graph: &mut Forward2BackwardGraphConverter,
            ) -> BackwardRecordedOpsBoxed<B::TensorPrimitive<D>> {
                Box::new(BackwardCatOps::<D, B>::new(
                    self.nodes
                        .iter()
                        .map(|node| {
                            let ops: BackwardNode<B::TensorPrimitive<D>> =
                                BackwardNode::from_node(node, graph);
                            Arc::new(ops)
                        })
                        .collect(),
                    self.dim,
                ))
            }
        }

        impl<const D: usize, B: Backend> BackwardRecordedOps<B::TensorPrimitive<D>>
            for BackwardCatOps<D, B>
        {
            fn backward_step(&self, state: &BackwardNodeState<B::TensorPrimitive<D>>) {
                let grad = state.grad();
                let indexes: Vec<_> = B::shape(&grad).dims.iter().map(|v| 0..*v).collect();
                let indexes: [std::ops::Range<usize>; D] = indexes.try_into().unwrap();

                self.nodes.iter().enumerate().for_each(|(i, node)| {
                    let mut indexes = indexes.clone();
                    indexes[self.dim] = i..i + 1;
                    node.state.update_grad(B::index(&grad, indexes));
                });
            }

            fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
                self.nodes
                    .iter()
                    .map(|node| {
                        let ops: RecordedOpsParentRef = node.clone();
                        ops
                    })
                    .collect()
            }
        }

        let nodes: Vec<_> = tensors.iter().map(|t| t.node.clone()).collect();
        let order = nodes.iter().map(|node| node.order).max().unwrap() + 1;

        let tensors_inner: Vec<B::TensorPrimitive<D>> =
            tensors.iter().map(|a| a.tensor()).collect();

        let out = B::cat(&tensors_inner, dim);

        let shape = *B::shape(&out);
        let state = crate::graph::node::ForwardNodeState::new(out);

        let ops = ForwardCatOps::<D, B>::new(nodes, dim);
        let ops = Box::new(ops);

        let node = crate::graph::node::ForwardNode::new(order, state, ops);
        let node = std::sync::Arc::new(node);

        ADTensor { node, shape }
    }

    fn relu<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        #[derive(Default, Debug)]
        struct Backward<B: Backend, const D: usize> {
            _b: B,
        }

        impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
            for Backward<B, D>
        {
            fn partial(
                &self,
                state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
            ) -> B::TensorPrimitive<D> {
                let zero = 0.to_elem();
                let mask = B::lower_equal_scalar(&state.output.value(), &zero);
                B::mask_fill(&state.output.grad(), &mask, zero)
            }
        }

        let output = B::relu(tensor.tensor_ref());
        let ops = Backward::<B, D>::default();

        unary_ops_wrapper(tensor.node.clone(), output, ops)
    }
}
