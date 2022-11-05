use super::{binary_ops_wrapper, unary_ops_wrapper};
use crate::{
    backend::{
        autodiff::{ADBackendDecorator, ADTensor},
        Backend,
    },
    graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState},
    ops::{TensorOps, TensorOpsNeg},
    Data, Shape,
};

impl<B: Backend, const D: usize> std::ops::Add<ADTensor<D, B>> for ADTensor<D, B> {
    type Output = ADTensor<D, B>;

    fn add(self, rhs: Self) -> Self::Output {
        ADBackendDecorator::add(&self, &rhs)
    }
}

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
        state.output.grad().neg()
    }
}

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
        let output = B::add(lhs.tensor_ref(), rhs.tensor_ref());
        let ops = AddBackward::<B, D>::default();

        binary_ops_wrapper(lhs.node.clone(), rhs.node.clone(), output, ops)
    }

    fn add_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        let output = B::add_scalar(lhs.tensor_ref(), rhs);
        let ops = AddScalarBackward::<B, D>::default();

        unary_ops_wrapper(lhs.node.clone(), output, ops)
    }

    fn sub<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        let output = B::sub(lhs.tensor_ref(), rhs.tensor_ref());
        let ops = SubBackward::<B, D>::default();

        binary_ops_wrapper(lhs.node.clone(), rhs.node.clone(), output, ops)
    }

    fn sub_scalar<const D: usize>(
        lhs: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
        rhs: &<ADBackendDecorator<B> as Backend>::Elem,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        let output = B::sub_scalar(lhs.tensor_ref(), rhs);
        let ops = SubScalarBackward::<B, D>::default();

        unary_ops_wrapper(lhs.node.clone(), output, ops)
    }
}
