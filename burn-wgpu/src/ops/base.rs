use std::{marker::PhantomData, sync::Arc};

use burn_tensor::{backend::Backend, Data, Shape};

use crate::{element::WGPUElement, pool::get_context, tensor::WGPUTensor, GraphicsAPI, WGPUDevice};

pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type Device<B> = <B as Backend>::Device;

pub type FloatTensor<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

pub type IntElem<B> = <B as Backend>::IntElem;
pub type IntTensor<B, const D: usize> = <B as Backend>::IntTensorPrimitive<D>;
pub type BoolTensor<B, const D: usize> = <B as Backend>::BoolTensorPrimitive<D>;

pub struct BaseOps<G: GraphicsAPI> {
    _g: PhantomData<G>,
}

impl<G: GraphicsAPI> BaseOps<G> {
    pub fn from_data<E: WGPUElement, const D: usize>(
        data: Data<E, D>,
        device: &WGPUDevice,
    ) -> WGPUTensor<E, D> {
        let context = get_context::<G>(device);
        let buffer = context.create_buffer_with_data(E::as_bytes(&data.value));

        WGPUTensor::new(context, data.shape, Arc::new(buffer))
    }

    pub fn to_data<E: WGPUElement, const D: usize>(tensor: &WGPUTensor<E, D>) -> Data<E, D> {
        let bytes = tensor.context.buffer_to_data(&tensor.buffer);
        let values = E::from_bytes(&bytes);

        Data::new(values.to_vec(), tensor.shape.clone())
    }

    pub fn to_device<E: WGPUElement, const D: usize>(
        tensor: WGPUTensor<E, D>,
        device: &WGPUDevice,
    ) -> WGPUTensor<E, D> {
        if &tensor.context.device == device {
            return tensor;
        }

        let context = get_context::<G>(device);
        tensor.to_context(context)
    }

    pub fn empty<E: WGPUElement, const D: usize>(
        shape: Shape<D>,
        device: &WGPUDevice,
    ) -> WGPUTensor<E, D> {
        let context = get_context::<G>(device);
        let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());

        WGPUTensor::new(context, shape, Arc::new(buffer))
    }
}
