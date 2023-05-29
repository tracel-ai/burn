use burn_tensor::Shape;
use std::sync::Arc;
use wgpu::Buffer;

use crate::context::Context;

#[derive(Debug, Clone)]
pub struct WGPUTensor<const D: usize> {
    pub(crate) context: Arc<Context>,
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) shape: Shape<D>,
    pub(crate) strides: [usize; D],
}

impl<const D: usize> WGPUTensor<D> {
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
        }
    }
    pub fn to_context(&self, context: Arc<Context>) -> Self {
        let data = self.context.buffer_to_data(&self.buffer);
        let buffer = Arc::new(context.create_buffer_with_data(&data));

        Self {
            context,
            buffer,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}
