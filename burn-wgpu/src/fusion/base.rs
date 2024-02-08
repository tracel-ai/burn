use super::{ElementWise, ElementWiseState};
use crate::{
    element::WgpuElement, fusion::ElementWiseBuilder, tensor::WgpuTensor, GpuBackend,
    JitGpuBackend, WgpuDevice,
};
use burn_compute::client::ComputeClient;
use burn_fusion::{client::MutexFusionClient, DeviceId, FusionBackend, FusionDevice};
use burn_tensor::Shape;
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

/// Fusion optimization type for WGPU.
///
/// More optimization variants should be added here.
pub enum WgpuOptimization<B: JitGpuBackend> {
    /// Element wise optimization.
    ElementWise(ElementWise<B>),
}

/// Fusion optimization state type for WGPU.
///
/// More optimization variants should be added here.
#[derive(Serialize, Deserialize)]
pub enum WgpuOptimizationState {
    /// Element wise state.
    ElementWise(ElementWiseState),
}

impl<B: JitGpuBackend> burn_fusion::Optimization<GpuBackend<B>> for WgpuOptimization<B> {
    fn execute(&mut self, context: &mut burn_fusion::stream::Context<'_, GpuBackend<B>>) {
        match self {
            Self::ElementWise(op) => op.execute(context),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::ElementWise(op) => op.len(),
        }
    }

    fn to_state(&self) -> WgpuOptimizationState {
        match self {
            Self::ElementWise(value) => WgpuOptimizationState::ElementWise(value.to_state()),
        }
    }

    fn from_state(device: &B::Device, state: WgpuOptimizationState) -> Self {
        match state {
            WgpuOptimizationState::ElementWise(state) => {
                Self::ElementWise(ElementWise::from_state(device, state))
            }
        }
    }
}

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

impl<B: JitGpuBackend> FusionBackend for GpuBackend<B> {
    type OptimizationState = WgpuOptimizationState;
    type Optimization = WgpuOptimization<B>;
    type FusionDevice = B::Device;
    type Handle = WgpuFusionHandle<B>;
    type FusionClient = MutexFusionClient<Self>;

    fn optimizations(
        device: B::Device,
    ) -> Vec<Box<dyn burn_fusion::OptimizationBuilder<Self::Optimization>>> {
        vec![Box::new(ElementWiseBuilder::new(device))]
    }

    fn float_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::FloatTensorPrimitive<D> {
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

    fn float_tensor_handle<const D: usize>(tensor: Self::FloatTensorPrimitive<D>) -> Self::Handle {
        tensor.into()
    }

    fn int_tensor_handle<const D: usize>(tensor: Self::IntTensorPrimitive<D>) -> Self::Handle {
        tensor.into()
    }

    fn bool_tensor_handle<const D: usize>(tensor: Self::BoolTensorPrimitive<D>) -> Self::Handle {
        tensor.into()
    }
}

pub fn strides_dyn_rank(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];

    let mut current = 1;
    shape.iter().enumerate().rev().for_each(|(index, val)| {
        strides[index] = current;
        current *= val;
    });

    strides
}

#[derive(new, Debug)]
/// Handle to be used when fusing operations.
pub struct WgpuFusionHandle<B: JitGpuBackend> {
    /// Compute client for wgpu.
    pub client: ComputeClient<B::Server, B::Channel>,
    /// The buffer where the data are stored.
    pub handle: burn_compute::server::Handle<B::Server>,
    /// The device of the current tensor.
    pub device: B::Device,
    pub(crate) strides: Vec<usize>,
}

impl<B: JitGpuBackend> Clone for WgpuFusionHandle<B> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            device: self.device.clone(),
            strides: self.strides.clone(),
        }
    }
}

unsafe impl<B: JitGpuBackend> Send for WgpuFusionHandle<B> {}
unsafe impl<B: JitGpuBackend> Sync for WgpuFusionHandle<B> {}

impl<B: JitGpuBackend> WgpuFusionHandle<B> {
    pub(crate) fn into_tensor<const D: usize, E: WgpuElement>(
        self,
        shape: Shape<D>,
    ) -> WgpuTensor<B, E, D> {
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

impl<B: JitGpuBackend, E: WgpuElement, const D: usize> From<WgpuTensor<B, E, D>>
    for WgpuFusionHandle<B>
{
    fn from(value: WgpuTensor<B, E, D>) -> Self {
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
    use crate::tests::TestJitGpuBackend;
    use burn_fusion::Fusion;

    pub type TestBackend = Fusion<GpuBackend<TestJitGpuBackend>>;
    pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> =
        burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;

    burn_tensor::testgen_all!();
    burn_autodiff::testgen_all!();
}
