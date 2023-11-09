use crate::{tensor::WgpuFusionHandle, FloatElement, GraphicsApi, IntElement, Wgpu, WgpuDevice};
use burn_fusion::{
    client::MutexFusionClient, graph::GreedyGraphExecution, DeviceId, FusedBackend, HandleDevice,
};
use burn_tensor::Shape;

impl HandleDevice for WgpuDevice {
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
impl<G, F, I> FusedBackend for Wgpu<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    type HandleDevice = WgpuDevice;
    type Handle = WgpuFusionHandle;
    type FusionClient = MutexFusionClient<Self, GreedyGraphExecution>;

    fn operations() -> Vec<Box<dyn burn_fusion::FusedOps<Self>>> {
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
