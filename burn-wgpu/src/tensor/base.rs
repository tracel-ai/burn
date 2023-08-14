use crate::unary;
use burn_tensor::Shape;
use std::{marker::PhantomData, sync::Arc};
use wgpu::Buffer;

use crate::{context::Context, element::WgpuElement, kernel::unary_default};

#[derive(Debug, Clone)]
pub struct WgpuTensor<E: WgpuElement, const D: usize> {
    pub(crate) context: Arc<Context>,
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) shape: Shape<D>,
    pub(crate) strides: [usize; D],
    elem: PhantomData<E>,
}

#[derive(Debug, Clone)]
pub struct WgpuTensorDyn<E: WgpuElement> {
    pub(crate) context: Arc<Context>,
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    elem: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> From<WgpuTensor<E, D>> for WgpuTensorDyn<E> {
    fn from(value: WgpuTensor<E, D>) -> Self {
        WgpuTensorDyn {
            context: value.context,
            buffer: value.buffer,
            shape: value.shape.dims.to_vec(),
            strides: value.strides.to_vec(),
            elem: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> From<WgpuTensorDyn<E>> for WgpuTensor<E, D> {
    fn from(value: WgpuTensorDyn<E>) -> Self {
        WgpuTensor {
            context: value.context,
            buffer: value.buffer,
            shape: Shape::new(value.shape.try_into().expect("Wrong dimension")),
            strides: value.strides.try_into().expect("Wrong dimension"),
            elem: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> WgpuTensor<E, D> {
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
            elem: PhantomData,
        }
    }

    pub fn to_context(&self, context: Arc<Context>) -> Self {
        let data = self.context.read_buffer(self.buffer.clone());
        let buffer = context.create_buffer_with_data(&data);

        Self {
            context,
            buffer,
            shape: self.shape.clone(),
            strides: self.strides,
            elem: PhantomData,
        }
    }
    pub fn can_mut_broadcast(&self, tensor_other: &WgpuTensor<E, D>) -> bool {
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

    pub fn copy(&self) -> Self {
        // Seems like using the copy buffer from the `wgpu` API leads to race condition when they
        // are used inplace afterward.
        //
        // To avoid them we need to execute the whole pipeline, which leads to significant
        // slowdowns.
        //
        // The solution is just to use a simple unary compute shader.
        unary!(CopyBuffer, body "output[id] = input[id];");
        unary_default::<CopyBuffer, E, D>(self.clone())
    }

    pub fn can_mut(&self) -> bool {
        if Arc::strong_count(&self.buffer) > 1 {
            return false;
        }

        true
    }

    pub fn assert_is_on_same_device(&self, other: &Self) {
        if self.context.device != other.context.device {
            panic!(
                "Both tensors should be on the same device {:?} != {:?}",
                self.context.device, other.context.device
            );
        }
    }

    pub fn is_contiguous(&self) -> bool {
        let mut current_stride = 0;
        for d in 0..D {
            let stride = self.strides[D - 1 - d];

            if stride < current_stride {
                return false;
            }

            current_stride = stride;
        }

        true
    }

    pub fn batch_swapped_with_row_col(&self) -> bool {
        for d in 0..D - 2 {
            let stride = self.strides[d];
            if stride < self.strides[D - 2] || stride < self.strides[D - 1] {
                return true;
            }
        }
        false
    }
}
