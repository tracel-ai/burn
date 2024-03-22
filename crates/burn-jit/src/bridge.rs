use crate::{kernel, tensor::JitTensor, FloatElement, JitBackend, Runtime};
use burn_tensor::{
    backend::BackendBridge,
    ops::{FloatElem, FloatTensor},
};
use core::marker::PhantomData;

/// Handle precision conversion for the jit backend.
#[derive(Debug)]
pub struct PrecisionBridge<R> {
    _runtime: PhantomData<R>,
}

impl<ROrigin, RTarget> BackendBridge<JitBackend<ROrigin>> for PrecisionBridge<RTarget>
where
    ROrigin: Runtime,
    RTarget: Runtime<
        Device = ROrigin::Device,
        Server = ROrigin::Server,
        Channel = ROrigin::Channel,
    >,
{
    type Target = JitBackend<RTarget>;

    fn into_target<const D: usize>(
        tensor: FloatTensor<JitBackend<ROrigin>, D>,
        device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target, D> {
        let tensor = kernel::cast::<
            ROrigin,
            FloatElem<JitBackend<ROrigin>>,
            FloatElem<JitBackend<RTarget>>,
            D,
        >(tensor);

        // The line bellow does the backend type cast.
        JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle)
    }

    fn from_target<const D: usize>(
        tensor: FloatTensor<Self::Target, D>,
        device: Option<burn_tensor::Device<JitBackend<ROrigin>>>,
    ) -> FloatTensor<JitBackend<ROrigin>, D> {
        let tensor = kernel::cast::<
            RTarget,
            FloatElem<JitBackend<RTarget>>,
            FloatElem<JitBackend<ROrigin>>,
            D,
        >(tensor);
        // The line bellow does the backend type cast.
        JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle)
    }
}
