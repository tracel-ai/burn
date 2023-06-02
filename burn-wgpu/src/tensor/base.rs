use burn_tensor::Shape;
use std::{marker::PhantomData, sync::Arc};
use wgpu::Buffer;

use crate::{context::Context, element::WGPUElement};

#[derive(Debug, Clone)]
pub struct WGPUTensor<E: WGPUElement, const D: usize> {
    pub(crate) context: Arc<Context>,
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) shape: Shape<D>,
    pub(crate) strides: [usize; D],
    elem: PhantomData<E>,
}

impl<E: WGPUElement, const D: usize> WGPUTensor<E, D> {
    pub fn new(context: Arc<Context>, shape: Shape<D>, buffer: Arc<Buffer>) -> Self {
        let mut strides = [0; D];

        let mut current = 1;
        shape
            .dims
            .iter()
            .enumerate()
            .rev()
            .for_each(|(index, val)| {
                strides[index] = current;
                current *= val;
            });

        Self {
            context,
            buffer,
            shape,
            strides,
            elem: PhantomData::default(),
        }
    }
    pub fn to_context(&self, context: Arc<Context>) -> Self {
        let data = self.context.buffer_to_data(&self.buffer);
        let buffer = Arc::new(context.create_buffer_with_data(&data));

        Self {
            context,
            buffer,
            shape: self.shape.clone(),
            strides: self.strides,
            elem: PhantomData::default(),
        }
    }
    pub fn can_mut_broadcast(&self, tensor_other: &WGPUTensor<E, D>) -> bool {
        if Arc::strong_count(&self.buffer) > 1 {
            return false;
        }

        for i in 0..D {
            // Output tensor will be different from the mutable tensor.
            if self.shape.dims[i] < tensor_other.shape.dims[i] {
                return false;
            }
        }

        true
    }

    pub fn can_mut(&self) -> bool {
        if Arc::strong_count(&self.buffer) > 1 {
            return false;
        }

        true
    }

    pub fn assert_is_on_save_device(&self, other: &Self) {
        if self.context.device != other.context.device {
            panic!(
                "Both tensors should be on the same device {:?} != {:?}",
                self.context.device, other.context.device
            );
        }
    }
}
