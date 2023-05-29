use burn_tensor::Shape;
use std::sync::Arc;
use wgpu::Buffer;

use crate::context::Context;

pub struct WGPUTensor<const D: usize> {
    context: Arc<Context>,
    buffer: Arc<Buffer>,
    shape: Shape<D>,
    strides: [usize; D],
}

impl<const D: usize> WGPUTensor<D> {
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
