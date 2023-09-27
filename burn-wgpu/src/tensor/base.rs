use crate::{
    compute::{WgpuComputeClient, WgpuHandle},
    unary, WgpuDevice,
};
use crate::{element::WgpuElement, kernel::unary_default};
use burn_tensor::Shape;
use std::marker::PhantomData;

/// The basic tensor primitive struct.
#[derive(Debug, Clone)]
pub struct WgpuTensor<E: WgpuElement, const D: usize> {
    /// Compute client for wgpu.
    pub client: WgpuComputeClient,
    /// The buffer where the data are stored.
    pub handle: WgpuHandle,
    /// The shape of the current tensor.
    pub shape: Shape<D>,
    /// The device of the current tensor.
    pub device: WgpuDevice,
    /// The strides of the current tensor.
    pub strides: [usize; D],
    elem: PhantomData<E>,
}

#[derive(Debug, Clone)]
pub(crate) struct WgpuTensorDyn<E: WgpuElement> {
    /// Compute client for wgpu.
    pub client: WgpuComputeClient,
    /// The buffer where the data are stored.
    pub handle: WgpuHandle,
    /// The device of the current tensor.
    pub device: WgpuDevice,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    elem: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> From<WgpuTensor<E, D>> for WgpuTensorDyn<E> {
    fn from(value: WgpuTensor<E, D>) -> Self {
        WgpuTensorDyn {
            client: value.client,
            handle: value.handle,
            device: value.device,
            shape: value.shape.dims.to_vec(),
            strides: value.strides.to_vec(),
            elem: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> From<WgpuTensorDyn<E>> for WgpuTensor<E, D> {
    fn from(value: WgpuTensorDyn<E>) -> Self {
        WgpuTensor {
            client: value.client,
            handle: value.handle,
            device: value.device,
            shape: Shape::new(value.shape.try_into().expect("Wrong dimension")),
            strides: value.strides.try_into().expect("Wrong dimension"),
            elem: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> WgpuTensor<E, D> {
    /// Create a new tensor.
    pub fn new(
        client: WgpuComputeClient,
        device: WgpuDevice,
        shape: Shape<D>,
        handle: WgpuHandle,
    ) -> Self {
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
            client,
            handle,
            shape,
            strides,
            device,
            elem: PhantomData,
        }
    }

    /// Change the context of the current tensor and return the newly transferred tensor.
    pub fn to_client(&self, client: WgpuComputeClient, device: WgpuDevice) -> Self {
        let bytes = self
            .client
            .read(&self.handle)
            .read_sync()
            .expect("Can only change client synchronously");
        let handle = client.create(&bytes);

        Self {
            client,
            handle,
            shape: self.shape.clone(),
            strides: self.strides,
            device,
            elem: PhantomData,
        }
    }

    pub(crate) fn can_mut_broadcast(&self, tensor_other: &WgpuTensor<E, D>) -> bool {
        if !self.handle.can_mut() {
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

    /// Copy the current tensor.
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

    /// Check if the tensor is safe to mutate.
    pub fn can_mut(&self) -> bool {
        self.handle.can_mut()
    }

    /// Assert that both tensors are on the same device.
    pub fn assert_is_on_same_device(&self, other: &Self) {
        if self.device != other.device {
            panic!(
                "Both tensors should be on the same device {:?} != {:?}",
                self.device, other.device
            );
        }
    }

    /// Check if the current tensor is contiguous.
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

    pub(crate) fn batch_swapped_with_row_col(&self) -> bool {
        for d in 0..D - 2 {
            let stride = self.strides[d];
            if stride < self.strides[D - 2] || stride < self.strides[D - 1] {
                return true;
            }
        }
        false
    }
}
