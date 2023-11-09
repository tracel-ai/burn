use crate::{
    compute::{WgpuComputeClient, WgpuHandle},
    element::WgpuElement,
    tensor::WgpuTensor,
    FloatElement, GraphicsApi, IntElement, Wgpu, WgpuDevice,
};
use burn_fusion::{
    client::MutexFusionClient, graph::GreedyGraphExecution, DeviceId, FusionBackend, FusionDevice,
};
use burn_tensor::Shape;
use core::marker::PhantomData;

impl FusionDevice for WgpuDevice {
    fn id(&self) -> DeviceId {
        match self {
            WgpuDevice::DiscreteGpu(index) => DeviceId::new(0, *index as u32),
            WgpuDevice::IntegratedGpu(index) => DeviceId::new(1, *index as u32),
            WgpuDevice::VirtualGpu(index) => DeviceId::new(2, *index as u32),
            WgpuDevice::Cpu => DeviceId::new(3, 0),
            WgpuDevice::BestAvailable => DeviceId::new(4, 0),
        }
    }
}

impl<G, F, I> FusionBackend for Wgpu<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    type FusionDevice = WgpuDevice;
    type Handle = WgpuFusionHandle;
    type FusionClient = MutexFusionClient<Self, GreedyGraphExecution>;

    fn operations() -> Vec<Box<dyn burn_fusion::FusionOps<Self>>> {
        Vec::new()
    }

    fn float_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::TensorPrimitive<D> {
        handle.into_tensor(shape)
    }

    fn int_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::IntTensorPrimitive<D> {
        handle.into_tensor(shape)
    }

    fn bool_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::BoolTensorPrimitive<D> {
        handle.into_tensor(shape)
    }

    fn float_tensor_handle<const D: usize>(tensor: Self::TensorPrimitive<D>) -> Self::Handle {
        tensor.into()
    }

    fn int_tensor_handle<const D: usize>(tensor: Self::IntTensorPrimitive<D>) -> Self::Handle {
        tensor.into()
    }

    fn bool_tensor_handle<const D: usize>(tensor: Self::BoolTensorPrimitive<D>) -> Self::Handle {
        tensor.into()
    }
}

#[derive(Debug, Clone)]
/// Handle to be used when fusing operations.
pub struct WgpuFusionHandle {
    /// Compute client for wgpu.
    pub client: WgpuComputeClient,
    /// The buffer where the data are stored.
    pub handle: WgpuHandle,
    /// The device of the current tensor.
    pub device: WgpuDevice,
    pub(crate) strides: Vec<usize>,
}

impl WgpuFusionHandle {
    pub(crate) fn into_tensor<const D: usize, E: WgpuElement>(
        self,
        shape: Shape<D>,
    ) -> WgpuTensor<E, D> {
        WgpuTensor {
            client: self.client,
            handle: self.handle,
            device: self.device,
            shape,
            strides: self.strides.try_into().expect("Wrong dimension"),
            elem: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> From<WgpuTensor<E, D>> for WgpuFusionHandle {
    fn from(value: WgpuTensor<E, D>) -> Self {
        Self {
            client: value.client,
            handle: value.handle,
            device: value.device,
            strides: value.strides.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_fusion::Fusion;

    pub type TestBackend = Fusion<Wgpu>;
    pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;

    burn_tensor::testgen_all!();
    burn_autodiff::testgen_all!();
}
