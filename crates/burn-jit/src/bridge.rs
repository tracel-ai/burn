use crate::{
    kernel, ops::to_device, tensor::JitTensor, FloatElement, IntElement, JitBackend, JitRuntime,
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

impl<R, FOrigin, IOrigin, FTarget, ITarget> BackendBridge<JitBackend<R, FOrigin, IOrigin>>
    for PrecisionBridge<R, FTarget, ITarget>
where
    R: JitRuntime,
    FOrigin: FloatElement,
    IOrigin: IntElement,
    FTarget: FloatElement,
    ITarget: IntElement,
{
    type Target = JitBackend<R, FTarget, ITarget>;

    fn into_target(
        tensor: FloatTensor<JitBackend<R, FOrigin, IOrigin>>,
        device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target> {
        let tensor = kernel::cast::<
            R,
            FloatElem<JitBackend<R, FOrigin, IOrigin>>,
            FloatElem<JitBackend<R, FTarget, ITarget>>,
        >(tensor);

        // The line below does the backend type cast.
        let tensor =
            JitTensor::new_contiguous(tensor.client, tensor.device, tensor.shape, tensor.handle);

        if let Some(device) = &device {
            to_device(tensor, device)
        } else {
            tensor
        }
    }

    fn from_target(
        tensor: FloatTensor<Self::Target>,
        device: Option<burn_tensor::Device<JitBackend<R, FOrigin, IOrigin>>>,
    ) -> FloatTensor<JitBackend<R, FOrigin, IOrigin>> {
        let tensor = kernel::cast::<
            R,
            FloatElem<JitBackend<R, FTarget, ITarget>>,
            FloatElem<JitBackend<R, FOrigin, IOrigin>>,
        >(tensor);
        // The line below does the backend type cast.
        let tensor =
            JitTensor::new_contiguous(tensor.client, tensor.device, tensor.shape, tensor.handle);

        if let Some(device) = &device {
            to_device(tensor, device)
        } else {
            tensor
        }
    }
}
