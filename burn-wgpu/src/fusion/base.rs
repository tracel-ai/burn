use crate::{
    compute::{compute_client, WgpuComputeClient, WgpuHandle},
    element::WgpuElement,
    fusion::FloatElementWiseFusionOps,
    tensor::WgpuTensor,
    FloatElement, GraphicsApi, IntElement, Wgpu, WgpuDevice,
};
use burn_fusion::{
    client::MutexFusionClient, graph::GreedyGraphExecution, DeviceId, FusionBackend, FusionDevice,
};
use burn_tensor::{
    ops::{FloatElem, IntElem},
    Shape,
};
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
        vec![Box::new(FloatElementWiseFusionOps::default())]
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

    fn create_handle_float(
        values: Option<Vec<FloatElem<Self>>>,
        shape: &[usize],
        device: &WgpuDevice,
    ) -> Self::Handle {
        match values {
            Some(values) => create_handle_data::<G, FloatElem<Self>>(values, shape, device),
            None => create_handle_empty::<G, FloatElem<Self>>(shape, device),
        }
    }

    fn create_handle_int(
        values: Option<Vec<IntElem<Self>>>,
        shape: &[usize],
        device: &WgpuDevice,
    ) -> Self::Handle {
        match values {
            Some(values) => create_handle_data::<G, IntElem<Self>>(values, shape, device),
            None => create_handle_empty::<G, IntElem<Self>>(shape, device),
        }
    }

    fn create_handle_bool(
        values: Option<Vec<bool>>,
        shape: &[usize],
        device: &WgpuDevice,
    ) -> Self::Handle {
        match values {
            Some(values) => create_handle_data::<G, u32>(
                values.into_iter().map(|a| a as u32).collect(),
                shape,
                device,
            ),
            None => create_handle_empty::<G, u32>(shape, device),
        }
    }
}

fn create_handle_data<G: GraphicsApi, E: WgpuElement>(
    values: Vec<E>,
    shape: &[usize],
    device: &WgpuDevice,
) -> WgpuFusionHandle {
    let client = compute_client::<G>(device);
    let buffer = client.create(E::as_bytes(&values));
    let strides = dyn_strides(&shape);

    WgpuFusionHandle::new(client, buffer, device.clone(), strides)
}

fn create_handle_empty<G: GraphicsApi, E: WgpuElement>(
    shape: &[usize],
    device: &WgpuDevice,
) -> WgpuFusionHandle {
    let client = compute_client::<G>(device);
    let mut num_elems = 1;
    for s in shape {
        num_elems *= s;
    }
    let buffer = client.empty(core::mem::size_of::<E>() * num_elems);
    let strides = dyn_strides(&shape);

    WgpuFusionHandle::new(client, buffer, device.clone(), strides)
}

fn dyn_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];

    let mut current = 1;
    shape.iter().enumerate().rev().for_each(|(index, val)| {
        strides[index] = current;
        current *= val;
    });

    strides
}

#[derive(new, Debug, Clone)]
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
