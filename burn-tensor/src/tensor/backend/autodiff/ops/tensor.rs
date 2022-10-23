use super::unary_ops_wrapper;
use crate::{
    backend::{
        autodiff::{ADBackendDecorator, ADTensor},
        Backend,
    },
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    ops::TensorOps,
    Data, Shape,
};

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
}
