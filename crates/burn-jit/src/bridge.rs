use crate::{
    kernel, ops::to_device, tensor::JitTensor, FloatElement, IntElement, JitBackend, Runtime,
};
use burn_tensor::{
    backend::BackendBridge,
    ops::{FloatElem, FloatTensor},
};
use core::marker::PhantomData;

/// Handle precision conversion for the jit backend.
#[derive(Debug)]
pub struct PrecisionBridge<R, F: FloatElement, I: IntElement> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
}

impl<ROrigin, FOrigin, IOrigin, RTarget, FTarget, ITarget>
    BackendBridge<JitBackend<ROrigin, FOrigin, IOrigin>>
    for PrecisionBridge<RTarget, FTarget, ITarget>
where
    ROrigin: Runtime,
    FOrigin: FloatElement,
    IOrigin: IntElement,
    RTarget:
        Runtime<Device = ROrigin::Device, Server = ROrigin::Server, Channel = ROrigin::Channel>,
    FTarget: FloatElement,
    ITarget: IntElement,
{
    type Target = JitBackend<RTarget, FTarget, ITarget>;

    fn into_target<const D: usize>(
        tensor: FloatTensor<JitBackend<ROrigin, FOrigin, IOrigin>, D>,
        device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target, D> {
        let tensor = kernel::cast::<
            ROrigin,
            FloatElem<JitBackend<ROrigin, FOrigin, IOrigin>>,
            FloatElem<JitBackend<RTarget, FOrigin, IOrigin>>,
            D,
        >(tensor);

        // The line below does the backend type cast.
        let tensor = JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);

        if let Some(device) = &device {
            to_device(tensor, device)
        } else {
            tensor
        }
    }

    fn from_target<const D: usize>(
        tensor: FloatTensor<Self::Target, D>,
        device: Option<burn_tensor::Device<JitBackend<ROrigin, FOrigin, IOrigin>>>,
    ) -> FloatTensor<JitBackend<ROrigin, FOrigin, IOrigin>, D> {
        let tensor = kernel::cast::<
            RTarget,
            FloatElem<JitBackend<RTarget, FTarget, ITarget>>,
            FloatElem<JitBackend<ROrigin, FOrigin, IOrigin>>,
            D,
        >(tensor);
        // The line below does the backend type cast.
        let tensor = JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);

        if let Some(device) = &device {
            to_device(tensor, device)
        } else {
            tensor
        }
    }
}
